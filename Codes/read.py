def readFASTAs(fileName):

    '''
    :param fileName:
    :return: genome sequences
    '''
    with open(fileName, 'r') as file:
        v = []
        genome = ''
        for line in file:
            if line[0] != '>':
                genome += line.strip()
            else:
                v.append(genome)
                genome = ''
        v.append(genome)
        del v[0]
        return v

def readLabels(fileName):

    '''
    :param fileName:
    :return: label of genome sequences
    '''
    with open(fileName, 'r') as file:
        v = []
        for line in file:
            if line != '\n':
                v.append((line.replace('\n', '')).replace(' ', ''))
        return v



def fetchXY(FASTAs, Labels):
    # print('Please, enter the full path of FASTA file:')
    # X = readFASTA(input().strip())
    #
    # print('Please, enter the full path of label file:')
    # Y = readLabel(input().strip())
    #
    # from sklearn.preprocessing import LabelEncoder
    # Y = LabelEncoder().fit_transform(Y)
    #
    # assert len(X)==len(Y), 'Numbers of FASTA and numbers of type are not equal.'

    X = readFASTAs(FASTAs)
    Y = readLabels(Labels)

    from sklearn.preprocessing import LabelEncoder
    Y = LabelEncoder().fit_transform(Y)

    assert len(X)==len(Y), 'Numbers of sequence and number of labels are not equal.'

    return X, Y




