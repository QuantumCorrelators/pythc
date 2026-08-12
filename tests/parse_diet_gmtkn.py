import yaml

from tests.reaction import Reaction, MolInfo, Educt, Product

KCALPERMOL_PER_HARTREE = 627.509_474

def parse_yaml_to_reactions(file: str) -> list[Reaction]:
    yaml_content = ""
    with open(file, 'r') as f:
        yaml_content = f.read()

    data = yaml.safe_load(yaml_content)
    parsed_reactions = []

    for dataset_name, reactions in data.items():
        if not reactions:
            continue

        for reaction_id, reaction_data in reactions.items():
            all_elements = set()
            energy = float(reaction_data.get('Energy', 0.0))
            out_weight = int(float(reaction_data.get('Weight', 0.0)))

            educts = []
            products = []
            educt_strs = []
            product_strs = []

            species_dict = reaction_data.get('Species', {})
            for species_name, species_data in species_dict.items():
                count = int(species_data.get('Count', 0))
                charge = int(species_data.get('Charge', 0))
                spin = int(species_data.get('UHF', 0))

                elements = species_data.get('Elements', [])
                for el in elements:
                    all_elements.add(el)

                positions = species_data.get('Positions', [])
                geom_lines = []
                for el, pos in zip(elements, positions):
                    geom_lines.append(f"{el:2s} {pos[0]:>10.5f} {pos[1]:>10.5f} {pos[2]:>10.5f}")
                geom_str = "\n".join(geom_lines)

                mol_info = MolInfo(
                    filename=species_name,
                    geom=geom_str,
                    spin=spin,
                    charge=charge,
                    n_atoms=len(elements),
                )

                # Assigning the exact count directly to the weight attribute
                if count < 0:
                    educts.append(Educt(weight=count, mol=mol_info))
                    educt_strs.append(f"{abs(count)} {species_name}")
                elif count > 0:
                    products.append(Product(weight=count, mol=mol_info))
                    product_strs.append(f"{count} {species_name}")

            reaction_str = f"{' + '.join(educt_strs)} -> {' + '.join(product_strs)}"

            reaction_obj = Reaction(
                dataset=dataset_name,
                file_name=reaction_str,
                reaction=reaction_id,
                products=products,
                educts=educts,
                energy_diff_kcalmol=energy,
                out_weight=out_weight,
                elements=all_elements
            )
            parsed_reactions.append(reaction_obj)

    return parsed_reactions