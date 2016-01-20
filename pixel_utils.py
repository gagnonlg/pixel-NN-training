def load_data(path,targets,shape=None):
    if shape is not None:
        data = np.zeros(shape)
        header = None
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            fst = True
            for i,row in enumerate(reader):
                if fst:
                    header = row
                    fst = False
                    continue
                data[i-1,:] = np.array(row)
    else:
        data = np.loadtxt(path,delimiter=',',skiprows=1)

    x = data[:,:-targets]
    y = data[:,-targets:]

    return x,y,header
