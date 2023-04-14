def make_lightcone(filenames, z_low=None, z_high=None, file_redshifts=None, cbin_bits=32, cbin_order='c', los_axis=0, raw_density=False, interpolation='linear', reading_function=None, box_length_mpc=None):
    '''
    Make a lightcone from xfrac, density or dT data. Replaces freq_box.
    
    Parameters:
        filenames (string or array): The coeval cubes. 
            Can be either any of the following:
            
                - An array with the file names
                
                - A text file containing the file names
                
                - The directory containing the files (must only contain one type of files)
        z_low (float): the lowest redshift. If not given, the redshift of the 
            lowest-z coeval cube is used.
        z_high (float): the highest redshift. If not given, the redshift of the 
            highest-z coeval cube is used.
        file_redshifts (string or array): The redshifts of the coeval cubes.
            Can be any of the following types:
            
            - None: determine the redshifts from file names
             
            - array: array containing the redshift of each coeval cube
            
            - filename: the name of a data file to read the redshifts from
            
        cbin_bits (int): If the data files are in cbin format, you may specify 
            the number of bits.
        cbin_order (char): If the data files are in cbin format, you may specify 
            the order of the data.
        los_axis (int): the axis to use as line-of-sight for the coeval cubes
        raw_density (bool): if this is true, and the data is a 
            density file, the raw (simulation units) density will be returned
            instead of the density in cgs units
        interpolation (string): can be 'linear', 'step', 'sigmoid' or
            'step_cell'. 
            Determines how slices in between output redshifts are interpolated.
    Returns:
        (lightcone, z) tuple
        
        - lightcone is the lightcone volume where the first two axes have the same size as the input cubes
        
        - z is an array containing the redshifts along the line-of-sight
        
    .. note::
        If z_low is given, that redshift will be the lowest one included,
        even if there is no coeval box at exactly that redshift. This can 
        give results that are subtly different from results calculated with
        the old freq_box routine.
    '''
    
    if not interpolation in ['linear', 'step', 'sigmoid', 'step_cell']:
        raise ValueError('Unknown interpolation type: %s' % interpolation)
    
    if reading_function is None:
        #Figure out output redshifts, file names and size of output
        filenames = _get_filenames(filenames)
        file_redshifts = _get_file_redshifts(file_redshifts, filenames)
        mesh_size = get_mesh_size(filenames[0])
    else:
        assert file_redshifts is not None
        mesh_size = reading_function(filenames[0]).shape

    assert len(file_redshifts) == len(filenames)
    
    output_z = _get_output_z(file_redshifts, z_low, z_high, mesh_size[0], box_length_mpc=box_length_mpc)

    #Make the output 32-bit to save memory 
    lightcone = np.zeros((mesh_size[0], mesh_size[1], len(output_z)), dtype='float32')
    
    comoving_pos_idx = 0
    z_bracket_low = None; z_bracket_high = None
    data_low = None; data_high = None
    
    # Make the lightcone, one slice at a time
    # print_msg('Making lightcone between %f < z < %f' % (output_z.min(), output_z.max()))
    print('Making lightcone between %f < z < %f' % (output_z.min(), output_z.max()))
    time.sleep(1)
    for ii in tqdm(range(len(output_z))):
        z = output_z[ii]
        z_bracket_low_new = file_redshifts[file_redshifts <= z].max()
        z_bracket_high_new = file_redshifts[file_redshifts > z].min()
        
        #Do we need a new file for the low z?
        if z_bracket_low_new != z_bracket_low:
            z_bracket_low = z_bracket_low_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_low))
            if data_high is None:
                if reading_function is None: data_low, datatype = get_data_and_type(filenames[file_idx], cbin_bits, cbin_order, raw_density)
                else: data_low = reading_function(filenames[file_idx])#; print('yes')
            else: #No need to read the file again
                data_low = data_high
            
        #Do we need a new file for the high z?
        if z_bracket_high_new != z_bracket_high:
            z_bracket_high = z_bracket_high_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_high))
            if reading_function is None: data_high, datatype = get_data_and_type(filenames[file_idx], cbin_bits, cbin_order, raw_density)
            else: data_high = reading_function(filenames[file_idx])
        
        #Make the slice by interpolating, then move to next index
        data_interp = _get_interp_slice(data_high, data_low, z_bracket_high, \
                                    z_bracket_low, z, comoving_pos_idx, los_axis, interpolation)
        lightcone[:,:,comoving_pos_idx] = data_interp
        # print('%.2f %% completed.'%(100*(ii+1)/output_z.size))
        comoving_pos_idx += 1
    print('...done')
    return lightcone, output_z


# TODO: to modify based on the Ushuu data format
def make_velocity_lightcone(vel_filenames, dens_filenames, z_low = None, \
                            z_high = None, file_redshifts = None, los_axis = 0):
    '''
    Make a lightcone from velocity data. Since velocity files contain momentum
    rather than actual velocity, you must specify filenames for both velocity
    and density.
    
    Parameters:
        vel_filenames (string or array): The coeval velocity cubes. 
            Can be any of the following:
            
                - An array with the file names
                
                - A text file containing the file names
                
                - The directory containing the files (must only contain one type of files)
        dens_filenames (string or array): The coeval density cubes.
            Same format as vel_filenames.
        z_low (float): the lowest redshift. If not given, the redshift of the 
            lowest-z coeval cube is used.
        z_high (float): the highest redshift. If not given, the redshift of the 
            highest-z coeval cube is used.
        file_redshifts (string or array): The redshifts of the coeval cubes.
            Can be any of the following types:
            
            - None: determine the redshifts from file names
             
            - array: array containing the redshift of each coeval cube
            
            - filename: the name of a data file to read the redshifts from
            
        los_axis (int): the axis to use as line-of-sight for the coeval cubes
        
    Returns:
        (lightcone, z) tuple
        
        - lightcone is the lightcone volume where the first two axes have the same size as the input cubes
        
        - z is an array containing the redshifts along the line-of-sight
    '''
    
    dens_filenames = _get_filenames(dens_filenames)
    file_redshifts = _get_file_redshifts(file_redshifts, dens_filenames)
    vel_filenames = _get_filenames(vel_filenames)
    assert(len(file_redshifts) == len(vel_filenames))
    assert(len(vel_filenames) == len(dens_filenames))
    mesh_size = get_mesh_size(dens_filenames[0])
    
    output_z = _get_output_z(file_redshifts, z_low, z_high, mesh_size[0])

    lightcone = np.zeros((3, mesh_size[0], mesh_size[1], len(output_z)), dtype='float32')
    
    comoving_pos_idx = 0
    z_bracket_low = None; z_bracket_high = None
    
    print('Making velocity lightcone between %f < z < %f' % (output_z.min(), output_z.max()))
    time.sleep(1)
    for ii in tqdm(range(len(output_z))):
        z = output_z[ii]
        z_bracket_low_new = file_redshifts[file_redshifts <= z].max()
        z_bracket_high_new = file_redshifts[file_redshifts > z].min()
        
        if z_bracket_low_new != z_bracket_low:
            z_bracket_low = z_bracket_low_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_low))
            dfile = DensityFile(dens_filenames[file_idx])
            vel_file = VelocityFile(vel_filenames[file_idx])
            data_low = vel_file.get_kms_from_density(dfile)
            del dfile
            del vel_file
            
        if z_bracket_high_new != z_bracket_high:
            z_bracket_high = z_bracket_high_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_high))
            dfile = DensityFile(dens_filenames[file_idx])
            vel_file = VelocityFile(vel_filenames[file_idx])
            data_high = vel_file.get_kms_from_density(dfile)
            del dfile
            del vel_file
        
        data_interp = _get_interp_slice(data_high, data_low, z_bracket_high, \
                                    z_bracket_low, z, comoving_pos_idx, los_axis)
        lightcone[:,:,:,comoving_pos_idx] = data_interp
        
        comoving_pos_idx += 1
    print('...done')
    return lightcone, output_z

