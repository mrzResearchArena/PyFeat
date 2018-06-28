import argparse

def main(args):

    # print('Please, enter the sequenceType (DNA/RNA/PROT):')

    # seqType = args.sequenceType.upper()
    # F = open('sequenceType.txt', 'w')
    # F.write(seqType);
    # F.close()

    # if seqType == 'DNA' or seqType == 'RNA' or seqType == 'PROTEIN' or seqType == 'PROT':
    #     None
    # else:
    #     print('!-- Error --!')
    #     print('Please, type DNA/RNA/PROT.')
    #     return


    import read
    X, Y = read.fetchXY(args.fasta, args.label)

    print('\nDatasets fetching done.')
    ############################################################################

    import generateFeatures

    # F = open('kGap.txt', 'w')
    # F.write(str(args.kGap));
    # F.close()

    ############################################################################
    print('Features extraction begins. Be patient! The machine will take some time.')

    T = generateFeatures.gF(args, X, Y)
    X_train = T[:,:-1]
    Y_train = T[:,-1]

    print('Features extraction ends.')
    print('[Total extracted feature: {}]\n'.format(X_train.shape[1]))

    #############################################################################

    if args.fullDataset == 1:
        print('Converting (full) CSV is begin.')
        import save
        save.saveCSV(X_train, Y_train, 'full')
        print('Converting (full) CSV is end.')


    if args.testDataset == 1:
        print('Converting (test) CSV is begin.')
        import save
        save.saveCSV(X_train, Y_train, 'test')
        print('Converting (test) CSV is end.')

    # #############################################################################

    if args.optimumDataset == 1:
        print('\nFeatures selection begins. Be patient! The Machine will take some time.')
        import selectedImportantFeatures
        X_train = selectedImportantFeatures.importantFeatures(X_train, Y_train)
        print('Features selection ends.')
        print('[Total selected feature: {}]\n'.format(X_train.shape[1]))
        print('Converting (optimum) CSV is begin.')
        import save
        save.saveCSV(X_train, Y_train, 'optimum')
        print('Converting (optimum) CSV is end.')
        #############################################################################

if __name__ == '__main__':

    ######################
    # Adding Arguments
    #####################

    p = argparse.ArgumentParser(description='Features Geneation Tool from DNA, RNA, and Protein Sequences')

    p.add_argument('-seq', '--sequenceType', type=str, help='DNA/RNA/PROTEIN/PROT', default='DNA')

    p.add_argument('-fa', '--fasta', type=str, help='~/FASTA.txt')
    p.add_argument('-la', '--label', type=str, help='~/Labels.txt')

    p.add_argument('-kgap', '--kGap', type=int, help='(l,k,p)-mers', default=5)
    p.add_argument('-ktuple', '--kTuple', type=int, help='k=1 then (X), k=2 then (XX), k=3 then (XXX),', default=3)

    p.add_argument('-full', '--fullDataset', type=int, help='saved full dataset', default=0, choices=[0, 1])
    p.add_argument('-test', '--testDataset', type=int, help='saved test dataset', default=0, choices=[0, 1])
    p.add_argument('-optimum', '--optimumDataset', type=int, help='saved optimum dataset', default=0, choices=[0, 1])

    p.add_argument('-pseudo', '--pseudoKNC', type=int, help='Generate feature: X, XX, XXX, XXX', default=0, choices=[0, 1])
    p.add_argument('-zcurve', '--zCurve', type=int, help='x_, y_, z_', default=0, choices=[0, 1])
    p.add_argument('-gc', '--gcContent', type=int, help='GC/ACGT', default=0, choices=[0, 1])
    p.add_argument('-skew', '--cumulativeSkew', type=int, help='GC, AT', default=0, choices=[0, 1])
    p.add_argument('-atgc', '--atgcRatio', type=int, help='atgcRatio', default=0, choices=[0, 1])

    p.add_argument('-f11', '--monoMono', type=int, help='Generate feature: X_X', default=0, choices=[0, 1])
    p.add_argument('-f12', '--monoDi', type=int, help='Generate feature: X_XX', default=0, choices=[0, 1])
    p.add_argument('-f13', '--monoTri', type=int, help='Generate feature: X_XXX', default=0, choices=[0, 1])
    p.add_argument('-f21', '--diMono', type=int, help='Generate feature: XX_X', default=0, choices=[0, 1])
    p.add_argument('-f22', '--diDi', type=int, help='Generate feature: XX_XX', default=0, choices=[0, 1])
    p.add_argument('-f23', '--diTri', type=int, help='Generate feature: XX_XXX', default=0, choices=[0, 1])
    p.add_argument('-f31', '--triMono', type=int, help='Generate feature: XXX_X', default=0, choices=[0, 1])
    p.add_argument('-f32', '--triDi', type=int, help='Generate feature: XXX_XX', default=0, choices=[0, 1])

    args = p.parse_args()

    main(args)



