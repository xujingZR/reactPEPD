"""calculate the react-PEPD of L-M-A system"""

import os
import pickle
from time import time
import numpy as np
from tqdm import tqdm
from mp_api.client import MPRester
import periodictable

# visualization
import matplotlib.pyplot as plt

# analyze the coordination environment
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments

from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.core import Composition



def analyze_coordination(structure) -> dict:
    """
    Analyze the coordination environment of the structure.
    Args:
        structure: pymatgen.Structure
    Returns:
        the coordination environment of every atom in the structure
    """
    lgf = LocalGeometryFinder()
    lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=False)
    lgf.setup_structure(structure)
    lgf.setup_parameter("additional_condition", 1)
    se = lgf.compute_structure_environments()
    strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3, additional_condition=0)
    lse = LightStructureEnvironments.from_structure_environments(
        strategy=strategy, structure_environments=se
        )
    
    return lse.get_statistics()

def cal_reaction_energy(reactants:list, products:list, normalized_element:str) -> float:
    """
    Calculate the reaction energy of given reactents and products. To compare the reaction energy 
    of different reactions, the reaction energy is normalized to the element.

    Args:
        reactants(list): the entries of reactants
        products:(list): the entries of products
        normalized_element(str): the element to normalize the reaction energy. 
    
    Returns:
        reaction_energy(float): the reaction energy of the reaction, which is normalized to the element.
    """
    cr = ComputedReaction(reactants, products)
    cr.normalize_to_element(normalized_element)
    reaction_energy = cr.calculated_reaction_energy

    return reaction_energy

def get_t_formula_ip1(formula_ip1:list, ion_source:str, precursor:str, formua_form_energy:dict) -> str:
    """
    The formula with the lowest formation energy is selected as the target formula.

    Args:
        formula_ip1(list): the formula of reactants
        ion_source(str): the formula of ion source, eg: "Li2CO3"
        precursor(str): the formula of precursor, eg: "Co3O4"
        formua_form_energy(dict): the formation energy of the formula
    Returns:
        t_formula_ip1(str): the formula of the target formula
    """
    
    formation_energy_list = []
    for i in formula_ip1:
        for j in formua_form_energy:
            if j[0] == i:
                f = j[1]["form_e"]
                break
        formation_energy_list.append(f)
    t_formula_ip1 = formula_ip1[formation_energy_list.index(min(formation_energy_list))]

    return t_formula_ip1

def reactPEPD(L, M, A, L_source, M_source, api_key, root_dir):
    """
    Calculate the react-PEPD of L-M-A system.

    Args:
        L(str): the element of L, eg: "Li"
        M(str): the element of M, eg: "Co"
        A(str): the element of A, eg: "O"
        L_source(str): the formula of L source, eg: "Li2CO3"
        M_source(str): the formula of M source, eg: "Co3O4"
        api_key(str): the api key of the materials project
        root_dir(str): the root directory of the project
    """
    chemsys = f"{M}-{L}-{A}"
    root_dir = root_dir
    workdir = os.path.join(root_dir, f"{L}-{M}-{A}")

    print("Start to calculate the react-PEPD of L-M-A system.")
    print(f"==>The chemsys is {chemsys}.")
    print(f"==>The workdir is {workdir}.")

    # check the workdir exists
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    os.chdir(workdir)

    print("==>Start to query the data from the materials project(MP).")

    with MPRester(api_key) as mpr:
        entries = mpr.materials.summary.search(chemsys=chemsys)
    print(f"==>The number of structures is {len(entries)} in {L}-{M}-{A}.")

    print("==>Start to analyze the coordination environment.")
    # check the number of atoms in chemsys
    for e in entries:
        num = len(e.structure)
        if num > 60:
            entries.remove(e)
    try:
        with open("types_res.pkl", "rb") as f:
            types_res = pickle.load(f)
    except:
        # analyze the coordination environment
        types_res = []
        error_entry = []
        for entry in tqdm(entries):
            structure = entry.structure
            try:
                t1 = time()
                types = analyze_coordination(structure)
                t2 = time()
                if (t2 - t1) > 1000:
                    long_entry = entry
                    types = None
                    print("warning")
                else:
                    pass
            except:
                types = None
                error_entry.append(entry)
            types_res.append(types)
    
    with open("types_res.pkl", "wb") as f:
        pickle.dump(types_res, f)
    
    print(f"==> the results of coordination analysis are save as types_res.pkl .")

    formula = []
    for i in entries:
        formula.append(i.structure.reduced_formula)
    
    formula_1 = []
    types_res_1 = []
    entry_1 = []
    for i in range(len(formula)):
        if types_res[i] is None:
            continue
        else:
            try:
                condition = types_res[i]["fraction_atom_coordination_environments_present"][M]["O:6"] == 1
                if condition:
                    formula_1.append(formula[i])
                    types_res_1.append(types_res[i])
                    entry_1.append(entries[i])
                else:
                    continue
            except:
                pass
    
    print(f"==> the number of structures is {len(entry_1)} after filtering.")

    print("==>Start to analyze the stacking mode of oxygen.")

    # analyze the stacking mode of oxygen
    neigbor_num = []
    nn_finder = VoronoiNN(tol=0.1)
    for e in tqdm(entry_1):
        test_s = e.structure
        # extract the oxygen atoms
        t_a = AseAtomsAdaptor.get_atoms(test_s)
        t_o_a = t_a[t_a.get_atomic_numbers() == periodictable.elements.symbol(A).number]
        t_o_s = AseAtomsAdaptor.get_structure(t_o_a)
        # voronoi analysis
        neighbors = []
        for site in t_o_s:
            nn = nn_finder.get_nn_info(t_o_s, t_o_s.index(site))
            neighbors.append(len(nn))
        neigbor_num.append(neighbors)
    
    entry_2 = []
    formula_2 = []
    types_res_2 = []
    for i in range(len(entry_1)):
        if np.sum(np.array(neigbor_num[i]) != 12) == 0:
            entry_2.append(entry_1[i])
            formula_2.append(formula_1[i])
            types_res_2.append(types_res_1[i])
    
    print(f"==> the number of structures is {len(entry_2)} after filtering.")
    
    # the lowest formation energy of structure is selected in the same composition
    formua_form_energy = {}
    for i in range(len(formula_2)):
        atomic_numbers = np.array(entry_2[i].structure.atomic_numbers)
        num_L = np.sum(atomic_numbers == periodictable.elements.symbol(L).number)
        num_M = np.sum(atomic_numbers == periodictable.elements.symbol(M).number)
        num_A = np.sum(atomic_numbers == periodictable.elements.symbol(A).number)
        if formula_2[i] not in formua_form_energy.keys():
            formua_form_energy[formula_2[i]] = {
                "form_e": entry_2[i].formation_energy_per_atom,
                "entry": entry_2[i],
                "types": types_res_2[i],
                "ratio_MA": num_M/num_A,
                "ratio_LA": num_L/num_A
            }
        else:
            if entry_2[i].formation_energy_per_atom < formua_form_energy[formula_2[i]]["form_e"]:
                formua_form_energy[formula_2[i]] = {
                    "form_e": entry_2[i].formation_energy_per_atom,
                    "entry": entry_2[i],
                    "types": types_res_2[i],
                    f"ratio_MA": num_M/num_A,
                    f"ratio_LA": num_L/num_A
                }
            else:
                continue
    
    formua_form_energy = sorted(formua_form_energy.items(), key=lambda x: x[1]["ratio_MA"])


    ratio_MA = [i[1]["ratio_MA"] for i in formua_form_energy]
    ratio_LA = [i[1]["ratio_LA"] for i in formua_form_energy]
    form_e = [i[1]["form_e"] for i in formua_form_energy]
    t_formula = [i[0] for i in formua_form_energy]    


    fig1, ax1 = plt.subplots()
    ax1.scatter(ratio_LA, ratio_MA, c=form_e, cmap='coolwarm')
    ax1.set_xlabel(f"{L}/{A} ratio")
    ax1.set_ylabel(f"{M}/{A} ratio")
    fig1.savefig(f"ratio_{L}{M}{A}.png", dpi=500, transparent=True)

    print("==> The formation energy distribution of the ratio of M/A and L/A is saved as ratio_form_e.pkl .")
    print("==> The visualization result is saved as ratio_{L}{M}{A}.png .")

    with open("ratio_form_e.pkl", "wb") as f:
        pickle.dump([ratio_LA, ratio_MA, form_e], f)

    ratio_M_groups = list(set(ratio_MA))

    entry_formula = {}
    for e in entries:
        entry_formula[e.composition.reduced_formula] = e
    
    ref_entries = []
    with MPRester(api_key=api_key) as mpr:
        for i in entries:
            t_entry = mpr.get_entries(i.formula_pretty, compatible_only=True)
            e = 0
            for j in t_entry:
                if "R2SCAN" not in str(j):
                    if j.energy_per_atom < e:
                        e = j.energy_per_atom
                        t = j
                else:
                    continue
            ref_entries.append(t)

        for i in [L_source, "O2", "CO2"]:
            t_entry = mpr.get_entries(i, compatible_only=True)
            e = 0
            for j in t_entry:
                if "R2SCAN" not in str(j):
                    if j.energy_per_atom < e:
                        e = j.energy_per_atom
                        t = j
                else:
                    continue
            ref_entries.append(t)
    
    t_entry = mpr.get_entries(M_source, compatible_only=True)
    e = 0
    for j in t_entry:
        if "R2SCAN" not in str(j):
            if j.energy_per_atom < e:
                e = j.energy_per_atom
                t = j
        else:
            continue
    ref_entries.append(t)

    for e in ref_entries:
        entry_formula[e.composition.reduced_formula] = e
    
    ratio_M_groups.sort()
    ratio_M_groups.reverse()
    ratio_MA = np.array(ratio_MA)
    t_formula = np.array(t_formula)
    formula_i = t_formula[ratio_MA == ratio_M_groups[0]][0]
    t_formula_i_list = []
    t_formula_i_list.append(formula_i)
    for i in range(len(ratio_M_groups)-1):
        formula_ip1 = t_formula[ratio_MA == ratio_M_groups[i+1]]
        formula_i = get_t_formula_ip1(formula_ip1, L_source, M_source, formua_form_energy)
        t_formula_i_list.append(formula_i)
    
    print("==> start to calculate the reaction energy.")

    crs = []
    res = []
    # 计算反应能
    for i in range(len(t_formula_i_list)-1):
        reactants = [entry_formula[t_formula_i_list[i]]]
        products = [entry_formula[t_formula_i_list[i+1]]]
        rc = reactants[0].composition
        pc = products[0].composition

        num_A_r = rc.get_atomic_fraction(A)
        num_A_p = pc.get_atomic_fraction(A)
        num_M_r = rc.get_atomic_fraction(M)
        num_M_p = pc.get_atomic_fraction(M)
        num_L_r = rc.get_atomic_fraction(L)
        num_L_p = pc.get_atomic_fraction(L)

        ratio_LA_r = num_L_r/num_A_r
        ratio_LA_p = num_L_p/num_A_p

        if ratio_LA_r < ratio_LA_p:
            reactants.extend([entry_formula[L_source], entry_formula["O2"]])
            products.extend([entry_formula["CO2"]])
        else:
            reactants.extend([entry_formula[M_source]])
            products.extend([entry_formula["O2"]])

        cr = ComputedReaction(reactants, products)
        cr.normalize_to_element(M)
        reaction_energy = cr.calculated_reaction_energy
        crs.append(cr)
        res.append(reaction_energy)

    res.reverse()

    form = []
    for j in t_formula_i_list:
        for i in formua_form_energy:
            if i[0] == j:
                f = i[1]["form_e"]
                form.append(f)
    
    form_react = []
    for i in range(len(ratio_M_groups)-1):
        if res[i]>0:
            form_react.append([
                (ratio_M_groups[i] + ratio_M_groups[i+1])/2,
                form[i] + res[i],
            ])
    
    ratio_form = [[ratio_M_groups[i], form[i]] for i in range(len(form))]

    ratio_form = np.array(ratio_form)
    data = np.concatenate([np.array(ratio_form), np.array(form_react)])
    data = data[data[:, 0].argsort()]
    fig2, ax2 = plt.subplots()
    ax2.plot(data[:, 0], data[:, 1], c='green')
    ax2.scatter(ratio_form[:, 0], ratio_form[:, 1], marker='o')
    
    # save the data
    with open("valley.pkl", "wb") as f:
        pickle.dump([data, ratio_form], f)
    
    print("==> The valley of the reaction energy is saved as valley.pkl .")

    ax2.set_xlabel(f"{M}/{A} ratio")
    ax2.set_ylabel("Reaction energy + formation energy (eV)")
    fig2.savefig(f"vally_{L}{M}{A}.png", dpi=500, transparent=True)

    print("==> The visualization result is saved as vally_{L}{M}{A}.png .")

    sites = []
    for i in range(len(t_formula_i_list)):
        c = Composition(t_formula_i_list[i])
        num_L = c.get_atomic_fraction(L)
        num_M = c.get_atomic_fraction(M)
        num_A = c.get_atomic_fraction(A)

        num_MA = num_M/num_A
        num_LA = num_L/num_A

        sites.append([num_LA, num_MA])
    
    fig3, ax3 = plt.subplots()
    ax3.scatter(ratio_LA, ratio_MA, c=form_e, cmap='coolwarm')
    # plt.scatter(4/12, 5/12, color='red', marker='*', s=100)
    ax3.set_xlabel(f"{L}/{A} ratio")
    ax3.set_ylabel(f"{M}/{A} ratio")
    for i in range(len(sites)-1):
        ax3.arrow(sites[i][0], sites[i][1], sites[i+1][0]-sites[i][0], sites[i+1][1]-sites[i][1], color='black', head_width=0.01, head_length=0.01, ls="--")

    print("==> The path of the reaction is saved as ratio_{L}{M}{A}.png .")
    fig3.savefig(f"ratio_{L}{M}{A}.png", dpi=500, transparent=True)

    with open("t_formula_i_list.txt", "w") as f:
        for s in t_formula_i_list:
            f.write(s + "\n")
    print("==> The target formula list is saved as t_formula_i_list.txt .")