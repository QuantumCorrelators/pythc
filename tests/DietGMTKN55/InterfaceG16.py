import yaml
from  optparse import OptionParser

parser = OptionParser()
parser.add_option('--NFinal', type=int, default=30,
                  help="Number systems to select")
parser.add_option('--Mode', type='string', default='Analyse',
                  help="Number systems to select")
(Opts, args) = parser.parse_args()


def WriteSystems(NFinal = 150, Suff = ""):
    Y=yaml.load(open("GoodSamples/AllElements_%03d%s.yaml"%(NFinal,Suff)))
    IDs = []
    FC = open("GIF/ComboList_%d.txt"%(NFinal), "w")
    for Set in Y:
        for l in sorted(Y[Set]):
            ID0 = Set + "-" + "%d"%(l)

            Comment = ID0 + " : energy = %.4f"%(Y[Set][l]['Energy'])

            FC.write("%s:\n"%(ID0))
            FC.write("  Energy: %.4f\n"%(Y[Set][l]['Energy']))
            FC.write("  Weight: %.4f\n"%(Y[Set][l]['Weight']))
            FC.write("  Species:\n")
            SS = Y[Set][l]['Species']
            for P in SS:
                ID = ID0 + "-" + str(P)
                IDs += [ID]

                FC.write("    - { ID: \"%s\", Count: %d }\n"\
                            %(ID, SS[P]['Count']))


                C = SS[P]['Charge']
                D = SS[P]['UHF']+1

                F = open("GIF/%s.gif_"%(ID), "w")
                F.write("%s\n\n"%(ID))
                F.write("%d %d\n"%(C,D))
                for l in range(SS[P]['Number']):
                    x, y, z = tuple(SS[P]['Positions'][l])
                    F.write("%-4s %12.5f %12.5f %12.5f\n"\
                                %(SS[P]['Elements'][l], x, y, z))
                F.close()
    FC.close()

    F = open("GIF/FullList_%d.txt"%(NFinal), "w")
    for ID in list(set(IDs)):
        F.write("%s.gif_\n"%(ID))
    F.close()

def G16ExtractEnergies(FileName):
    # Extract the energy from a Gaussian run
    # Var = grep "SCF Done" FILENAME
    # Energy in Ha is 5th element after split
    # This assume a file processed like in the
    # examples
    F=open(FileName)
    E={}
    for L in F:
        if L[0]=="#":
            continue

        T=L.split()
        if len(T)>1:
            ID=T[0][:-5]
            En=float(T[1])
            E[ID]=En*627.509 # Ha -> kcal/mol
        else:
            print("Warning! %s not defined"%(T[0]))
    return E

def ReadSystems(NFinal=30, InputFile=None):
    Y=yaml.load(open("GIF/ComboList_%d.txt"%(NFinal)))

    if InputFile is None:
        InputFile = "Results\EnergyList_B3LYP_%d.txt"%(NFinal)

    G16Energy = G16ExtractEnergies(InputFile)

    WarningList = []

    MAE, MAEDen = 0., 0.
    Errors = {}
    for System in Y:
        Energy = Y[System]['Energy']
        Weight = Y[System]['Weight']

        EnergyError = False
        EnergyApprox = 0.
        for S in Y[System]['Species']:
            if not S['ID'] in G16Energy:
                EnergyError = True
            else:
                EnergyApprox += G16Energy[S['ID']] \
                    *float(S['Count'])

        EnergyDiff = EnergyApprox - Energy
        if EnergyError:
            print("Skipping system due to absence: %s [Weight: %.2f]"\
                  %(System, Weight))
            WarningList += [S['ID'] for S in Y[System]['Species']]
        elif abs( EnergyDiff*Weight )> 1e3:
            ErrList = [S['ID'] for S in Y[System]['Species']]
            print("Major error: %s - %.2f vs %.2f [Weight: %.2f]:"\
                  %(System, EnergyApprox, Energy, Weight))
            print("Species: "+", ".join(ErrList))
            print("Energies:"+", ".join(
                ["%.2f"%(G16Energy[x]) for x in ErrList] ))
            WarningList += ErrList
        elif not(EnergyError):
            print("%-16s %8.2f %8.2f %8.2f"%(System, EnergyApprox, Energy,
                                             abs( EnergyDiff))  )
            Errors[System] = EnergyDiff

            MAE += abs( EnergyDiff ) * Weight
            MAEDen += 1. # Weight

    if len(WarningList)>0: 
        print("The following systems probably have errors:")
        print("- [ %s ]"%( " ".join("%s"%W for W in WarningList)))

    return MAE/MAEDen, Errors

Mode = Opts.Mode.upper()[:2]
if Mode=="GE": # Generation mode - makes the .gif_ files
    WriteSystems(NFinal = Opts.NFinal)
else:
    MAE, Errors = ReadSystems(Opts.NFinal)
    print("NFinal = %3d, NActual = %3d, WTMAD2 = %.3f"\
          %(Opts.NFinal, len(list(Errors)), MAE))
