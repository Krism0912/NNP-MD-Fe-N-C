import logging

import torch
from ase.calculators.calculator import Calculator, all_changes
from PaiNN.data import AseDataReader
import numpy as np

class EnsembleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
            self,
            models,
            energy_scale=1.0,
            forces_scale=1.0,
            #        stress_scale=1.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.ase_data_reader = AseDataReader(self.cutoff)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale

        self.Nn=len(self.models)
        self.intermediates={}

        self.Bias_init_Args_Setting()


    def Bias_init_Args_Setting(self):
        self.beta_f = 0.1  # No use
        self.B = 0.3


        self.model_num = self.Nn

        self.alpha = "manual"
        self.alpha_factor = 0
        self.alpha_f = 0
        self.A = 8
        self.bias_step_min = 0
        self.bias_step_max = 100

        self.rate_e_a = 0
        self.rate_f_a = 0
        self.rate_e_b = 0
        self.rate_f_b = 0

        self.sigma_factor = 0

        self.step_to_start_ramping = 0
        self.ramping_interval_a = 0
        self.ramping_interval_b = 0

    def bias_Efunc_ex2(self,Na,sigma_e):
        self.beta_e=1/(2.0*Na*self.B**2)
        return self.A * (np.exp(-self.beta_e * sigma_e) - 1), self.beta_e * sigma_e

    def bias_Ffunc_ex2(self,sigma_e, sigma_grad_e):
        return self.A * self.beta_e * np.exp(-self.beta_e * sigma_e) * sigma_grad_e

    def compute_sigma_bias_potential(self, atoms, Efunc, Ffunc, enable_energy_bias=True, enable_force_bias=True,
                                     bias_atoms=False, epsilon=0.001, disable_nnp=False):
        if self.Nn < 2:
            print('Error: compute_sigma_bias_potential requires more than 1 network in the ensemble.')
            raise ValueError

        self.atoms=atoms.copy()
        self.Na = len(self.atoms)

        E, F = self.compute_props()
        
        E=np.array(E) #Unit eV
        F=np.array(F) #Unit eV/A

        # --------- ENERGY BIAS ---------
        if enable_energy_bias:

            E_v2 = np.sum(np.power(E - np.mean(E), 2.0)) / self.Nn # 2.0
            E_dx = (-2.0 / self.Nn) * np.sum((E - np.mean(E)).reshape(self.Nn, 1, 1) * (F - np.mean(F, axis=0)), axis=0)
        else:
            E_v2 = 0.0
            E_dx = 0.0 * np.sum(F, axis=0)

        # --------- FORCE BIAS ----------
        # dF and C
        if enable_force_bias:
            dF = F - np.mean(F, axis=0)[np.newaxis, :, :]
            eta = epsilon / np.mean(F, axis=0).max()

            # Forward calc
            self.atoms = atoms.copy()

            E1,F1=[],[]
            for idx,dFi in enumerate(dF):
                new_atoms=atoms.copy()
                new_pos=self.atoms.get_positions()[np.newaxis,:,:] + eta * dFi
                new_pos=np.squeeze(new_pos,0)
                new_atoms.set_positions(new_pos)
                Ei,Fi=self.compute_props(new_atoms,idx)
                E1.append(Ei)
                F1.append(Fi)
            E1 = np.array(E1)
            F1 = np.array(F1)

            # Backward calc
            E2,F2=[],[]
            for idx,dFi in enumerate(dF):
                new_atoms=atoms.copy()
                new_pos=self.atoms.get_positions()[np.newaxis,:,:] - eta * dFi
                new_pos=np.squeeze(new_pos,0)
                new_atoms.set_positions(new_pos)
                Ei,Fi=self.compute_props(new_atoms,idx)
                E2.append(Ei)
                F2.append(Fi)

            E2=np.array(E2)
            F2=np.array(F2)

            # Computer ds/dx via central difference
            F_dx = np.sum((F1 - F2) / (eta), axis=0) / (self.Nn * (self.Nn - 1))

            # Compute var sqr
            F_v2 = np.sum(np.sum(np.power(dF, 2), axis=0) / (self.Nn * (self.Nn - 1)))
        else:
            F_dx = 0.0 * np.sum(F, axis=0)
            F_v2 = 0.0

        E_bias, factor=self.bias_Efunc_ex2(self.Na,E_v2)

        # import logging
        # logging.info('E_bias={} sigmaF^2={} sigmaE^2={} factor={}'.format(E_bias,F_v2,E_v2,factor))

        print('E_bias=',E_bias,F_v2,E_v2,factor)
        F_bias = self.bias_Ffunc_ex2(E_v2, E_dx)


        if bias_atoms:
            bias_mask = np.array([[1.0 if x in bias_atoms else 0.0 for x in range(F_bias.shape[0])]])
            F_bias = F_bias * bias_mask.T

        E_nnp = np.mean(E, axis=0)
        F_nnp = np.mean(F, axis=0)

        print('F_bias=',np.linalg.norm(F_bias))
        print('F_nnp=',np.linalg.norm(F_nnp))
        print('r={}'.format(np.mean(np.linalg.norm(F_bias,axis=1)/np.linalg.norm(F_nnp,axis=1))))
        import sys
        sys.exit(0)

        self.intermediates = {'E_bias': E_bias, 'E_nnp': E_nnp, 'F_bias': F_bias, 'F_nnp': F_nnp, 'E_v2': E_v2,
                              'F_v2': F_v2,
                              'F_sig': np.sqrt(F_v2), 'E_sig': np.sqrt(E_v2) / np.sqrt(self.Na),
                              'energy_var': np.var(E),
                              'forces_var': np.var(np.stack(F), axis=0),
                              'energy_predict':E
                              }
        if not disable_nnp:
            return E_bias + E_nnp, F_bias + F_nnp, np.std(E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(F, axis=0))
        else:
            return E_bias, F_bias, np.std(E, axis=0) / np.sqrt(float(self.Na)), np.mean(np.std(F, axis=0))

    def compute_props(self, atoms=None,compute_model_idx=-1):
        model_inputs = self.ase_data_reader(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }
        predictions = {'energy': [], 'forces': []}
        if compute_model_idx == -1:
            if atoms is not None:
                self.atoms = atoms.copy()


            for model in self.models:
                model_results = model(model_inputs)
                predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
                predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)

            results = {"energy": predictions['energy']}

            results["forces"] = np.stack(predictions['forces'], axis=0)

            return results['energy'], results["forces"]
        else:
            model=self.models[compute_model_idx]
            model_results = model(model_inputs)
            # predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            # predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)
            # results = {"energy": predictions['energy']}
            # results["forces"] = np.stack(predictions['forces'], axis=0)
            return model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale, model_results["forces"].detach().cpu().numpy() * self.forces_scale


    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        self.bias_args = SET_BIAS_PARAMS()

        model_inputs = self.ase_data_reader(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        predictions = {'energy': [], 'forces': []}
        for model in self.models:
            model_results = model(model_inputs)
            predictions['energy'].append(model_results["energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)



        energy, forces, Estddev, Fstddev = self.compute_sigma_bias_potential(atoms,
                                                                           self.bias_args.Efunc,
                                                                           self.bias_args.Ffunc,
                                                                          True,
                                                                           False,
                                                                           False,
                                                                           0.001,
                                                                           False)

        results = {"energy":energy,"forces":forces}


        ensemble={
            "E_stddev": Estddev,
            "F_stddev": Fstddev,
            "E_bias":self.intermediates['E_bias'],
            "E_nnp": self.intermediates['E_nnp'],
        }

        results['ensemble'] = ensemble

        self.results = results


class Bias_init_Args_Setting:
    def __init__(self):
        self.beta_f= 0.1 #No use
        self.beta_e = 0.3
    
        self.model_num= 4
    
        self.alpha="manual"
        self.alpha_factor=0
        self.alpha_f = 0
        self.alpha_e = 8
        self.bias_step_min=0
        self.bias_step_max=100
    
        self.rate_e_a=0
        self.rate_f_a=0
        self.rate_e_b=0
        self.rate_f_b=0
    
        self.sigma_factor=0
    
        self.step_to_start_ramping=0
        self.ramping_interval_a=0
        self.ramping_interval_b=0




class SET_BIAS_PARAMS:
    def __init__(self,atoms=None,ml=None):
        if not atoms:
            self.atoms=atoms
            NA = 10
        else:
            NA=len(atoms)

        args=Bias_init_Args_Setting()
        self.beta_f = 1 / (2 * (args.beta_f* (2 * np.sqrt(NA))) ** 2)
        self.beta_e = 1 / (2 * (args.beta_e * (
                    2 * np.sqrt(NA))) ** 2)  ##1 / ( 2 * (beta_e*(2*np.sqrt(10))/27.21138505)**2 )
        self.beta_e = 1 / (2*args.model_num *args.beta_e**2)
        if args.alpha == "auto":
            self.weight_f = args.alpha_factor * (
                        self.beta_f * ml["uqmax"]["fuqmax"] * ml["uqmax"]["fuqmax"] + ml["uqmax"]["fuqmax"]) / self.beta_f
            self.weight_e = args.alpha_factor * (
                        self.beta_e * ml["uqmax"]["euqmax"] * ml["uqmax"]["euqmax"] + ml["uqmax"]["euqmax"]) / self.beta_e
        elif args.alpha == "manual":
            self.weight_f = args.alpha_f
            self.weight_e = args.alpha_e
        else:
            self.weight_f = args.alpha_factor * (
                    self.beta_f * ml["uqmax"]["fuqmax"] * ml["uqmax"]["fuqmax"] + ml["uqmax"]["fuqmax"]) / self.beta_f
            self.weight_e = args.alpha_factor * (
                    self.beta_e * ml["uqmax"]["euqmax"] * ml["uqmax"]["euqmax"] + ml["uqmax"]["euqmax"]) / self.beta_e


        # these uncertainties are used in md while loop to check if the criteria met
        self.E_variance = 0
        self.F_variance = 0
        self.E_variance_std = 0
        self.F_variance_std = 0

        # used in md while loop to check if the maximum step achieved
        self.step_iteration = 0

        # used in alpha and beta ramping up to check the interval
        self.counter_to_iterate_a = 1
        self.counter_to_iterate_b = 1

        # set the random number of initial non-bias steps (between bias_step_min and bias_step_max)
        self.bias_step = np.random.randint(args.bias_step_min,args.bias_step_max)


        # used to print highest achieved sigma at every step
        self.Fs_max = 0
        self.Es_max = 0
        self.Fs_Es_sum_max = 0

        # Bias - Exponent: a*exp(-b*x)
        def bias_Efunc_ex(sigma_f, sigma_e):
            return self.weight_f * np.exp(-self.beta_f * sigma_f) + self.weight_e * np.exp(-self.beta_e * sigma_e)

        def bias_Ffunc_ex(sigma_f, dsigma, sigma_e, sigma_grad_e):
            return self.weight_f * self.beta_f * np.exp(-self.beta_f * sigma_f) * dsigma + \
                self.weight_e * self.beta_e * np.exp(-self.beta_e * sigma_e) * sigma_grad_e

        #Bias - article
        def bias_Efunc_ex2(sigma_f, sigma_e):
            return  self.weight_e * (np.exp(-self.beta_e * sigma_e)-1),self.beta_e * sigma_e

        def bias_Ffunc_ex2(sigma_f, dsigma, sigma_e, sigma_grad_e):
            return self.weight_e * self.beta_e * np.exp(-self.beta_e * sigma_e) * sigma_grad_e


        self.Efunc=bias_Efunc_ex2
        self.Ffunc=bias_Ffunc_ex2
