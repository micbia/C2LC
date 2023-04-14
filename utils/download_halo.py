import numpy as np, os

typ_sim = 'Uchuu'
#typ_sim = 'MiniUchuu'
typ_cat = 'RockstarExtended'

# see: http://skiesanduniverses.org/Simulations/Uchuu/UchuuDR1Products/
wabpage = 'http://skun.iaa.es/SUsimulations/UchuuDR1/'
path_download = webpage+'/'+typ_sim+'/'+typ_cat
path_out = '/work/ska/HIRAX/UchuuSim/%s/%s/' %(typ_sim, typ_cat)

try:
    os.mkdir(path_out)
except:
    print(' %s catalog already exist: SKIP' %typ_cat)

# read redshift of Ushuu simulation
idx_dir, redshift = np.loadtxt(path_out+'ushuu_redshift.txt', unpack=True, usecols=(0, 1))

# get catalogue according to HIRAX redshift range
#idx_dir = np.array(idx[(redshift > 0.8) * (redshift < 2.5)], dtype=int)

idx_dir = [36, 20]

for i in idx_dir:
    str_z = ('%.2f' %redshift[i-1]).replace('.', 'p')

    if('Mini' in typ_sim):
        os.chdir(path_out)

        if not (os.path.exists('%sMiniUchuu_halolist_z%s.h5' %(path_out, str_z))):
            command_line = 'wget %sMiniUchuu_halolist_z%s.h5' %(path_download, str_z)
            os.system(command_line)
        else:
            print(' File MiniUchuu_halolist_z%s.h5 exist: SKIP' %(str_z))

    else:
        try:
            os.mkdir('%shalodir_%03d' %(path_out, i))
        except:
            pass
        
        os.chdir('%shalodir_%03d' %(path_out, i))
        
        for j in range(100):
            if not (os.path.exists('%shalodir_%03d/halolist_z%s_%d.h5' %(path_out, i, str_z, j))):
                command_line = 'wget %shalodir_%03d/halolist_z%s_%d.h5' %(path_download, i, str_z, j)
                os.system(command_line)
                #print(command_line)
            else:
                print(' File halodir_%03d/halolist_z%s_%d.h5 exist: SKIP' %(i, str_z, j))
