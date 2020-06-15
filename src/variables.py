import pathlib
import spherpro.db as db
import src

"""
Contains global variables such as column names,
naming templates and constants.
"""
class Vars:
    FOL_MODULE = pathlib.Path(src.__file__).absolute().parent
    FOL_PROJECT = FOL_MODULE.parent
    FOL_REPORT = FOL_PROJECT / 'reports'
    FOL_FIGURES = FOL_REPORT / 'figures'

    # From 99_site_overviews
    COL_MEASURETYPE = 'MeasurementType'
    COL_MEASURENAME = 'MeasurementName'
    COL_STACK = 'Stack'
    COL_OBJECTTYPE = 'ObjectType'
    COL_CHANNELIDX = 'ChannelIdx'
    COL_CHANNEL = 'MetalChannel'

    COL_XSLIDE_START = 'ROIStartXPosUm'
    COL_X_START = 'x_start'
    COL_YSLIDE_START ='ROIStartYPosUm'
    COL_YSLIDE_END ='ROIEndYPosUm'
    COL_Y_START = 'y_start'
    COL_ACROIID = 'AcquisitionROIID'

    COL_MAXX = 'MaxX'
    COL_MAXY = 'MaxY'

    COL_ACSESSION = 'AcSession'
    COL_ACID = 'AcquisitionID'
    DICT_PANNEL_COLS = {'MetalTag': COL_CHANNEL}
    DICT_IMAGE_COLS = {'Metadata_acsession': COL_ACSESSION,
                      'Metadata_acid': COL_ACID}

    SUFFIX_ACMETA = '_Acquisition_meta.csv'
    COL_MCD_ID = 'ID'
    VAL_TEMP = 'temp'
    VAL_IMAGE = 'image'
    VAL_INTENSITY = 'Intensity'

    COL_IMCAC = 'imc_ac'

    # From Variable Base Helper
    COL_CHANNELNAME = db.ref_planes.channel_name.key
    COL_CONDID = db.conditions.condition_id.key
    COL_CONDLEVEL = COL_CONDID + 'level'
    COL_CONDNAME = db.conditions.condition_name.key
    COL_FILTERVAL = db.object_filters.filter_value.key
    COL_GOODNAME = 'goodname'
    COL_IMGID = db.images.image_id.key
    COL_IMGLEVEL = COL_IMGID + 'level'
    COL_IMID = db.images.image_id.key
    COL_ISNB = 'isnb'
    COL_MEASID = db.measurements.measurement_id.key
    COL_MEASID = db.measurements.measurement_id.key
    COL_MEASNAME = db.measurement_names.measurement_name.key
    COL_MEASTYPE = db.measurements.measurement_type.key
    COL_OBJID = db.objects.object_id.key
    COL_SITEID = db.sites.site_id.key
    COL_SITELEVEL = COL_SITEID + 'level'
    COL_VALUE = db.object_measurements.value.key
    COL_WORKING = 'working'
    COL_D2RIM = 'distrim'
    COL_PLATEID = db.conditions.plate_id.key
    COL_PLATELEVEL = COL_PLATEID + 'level'
    
    # Filters
    FIL_MITOSIS = 'is_mitotic'
    FIL_APOPTOSIS = 'is_apoptotic' 
    
    # experiment layout
    COL_CELLLINE = 'cellline'
    COL_TIMEPOINT = 'time_point'
    COL_CONCENTRATION = 'concentration'
    
    # Default orders
    CAT_CELLLINE = ['HT29', 'DLD1', '293T', 'T47D']
    
    # Fit marker relations
    COL_GENE = 'gene'
    COL_GENE_UNTAGGED = 'gene_untagged'
    COL_TAG = 'tag'
    COL_DOXO = 'doxocycline'
    COL_DILUTION = 'dilution'
    COL_GOODNAME = 'goodname'
    COL_ISNB = 'isnb'
    COL_FITTED = 'fitted'
    COL_RESID = 'residual'
    COL_IMGID = db.images.image_id.key
    COL_OBJ_NR = db.objects.object_number.key
    COL_FC = 'fc'
    COL_P = 'p'
    COL_DF = 'DF'
    COL_DELTA = 'delta'
    COL_TSTAT = 't'
    COL_TAGSTAT = 'TagStat'
    COL_POSSTAT = 'PosStat'
    COL_WORKING = 'working'
    COL_N = 'n'
    COL_N_OVEREXPR = 'n_overexpr'
    COL_FC_CENS = 'fc_cens'
    COL_NB = 'nb'
    COL_P_CORR = 'p_corrected'
    COL_ISSIG = 'is_sig_sel'
    COL_FITCONDITIONNAME = 'FitConditionName'
    COL_VALUES = db.object_measurements.value.key

    COL_COEFNAME = 'coefname'