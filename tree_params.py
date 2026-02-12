# tree_params.py
from bionic_apps.ai import ClassifierType
from bionic_apps.bci_train_test_suites import CrossSubjectExpHandler
from bionic_apps.databases import Databases
from bionic_apps.databases.eeg.defaults import BOTH_LEGS, REST, RIGHT_HAND, SPATIAL, ROTATION, WORD, SUBTRACT, MUSIC, LEFT_HAND
from bionic_apps.feature_extraction import FeatureType
from bionic_apps.preprocess.io import SubjectHandle

LOG_DIR = 'bci_tests/tree'
TEST_NAME = 'EMB'

partition = 'gpu'
exclude = 'wald'
gpu_type = 1
cpu_cores = 12

test_class = CrossSubjectExpHandler

default_kwargs = dict(
    db_name=Databases.CYBATHLON2028_INIT,
    feature_type=FeatureType.RAW,
    epoch_tmin=0.5, epoch_tmax=5,
    window_len=1, window_step=.1,
    feature_kwargs={'norm': True},
    use_drop_subject_list=True,
    filter_params=dict(order=5, l_freq=.2, h_freq=45),
    do_artefact_rejection=False,
    balance_data=True,
    subject_handle=SubjectHandle.INDEPENDENT_DAYS,
    n_splits=5,

    classifier_type=ClassifierType.TREE_OF_EXPERTS_EEGNET,
    classifier_kwargs=dict(
        validation_split=.2,
        epochs=150,  #150
        patience=20,
        give_up=40,
        weight_classes=True,
        train_model=True, #true
        #learning_rate=1e-4,
        node_loss_weight=1,
    ),

    log_file='out.csv', base_dir='.',
    save_classifiers=True,
    subjects='all',
    tl_subjects='all',

    binarize_labels=False,
    selected_labels=[SPATIAL, REST, ROTATION, WORD, SUBTRACT, MUSIC, RIGHT_HAND, LEFT_HAND, BOTH_LEGS],
    #selected_labels=[REST, ROTATION, WORD, SUBTRACT, MUSIC, BOTH_LEGS],

    db_file='tmp/database_tree_6class.hdf5',
    fast_load=False,

    #Questo è il “root” dove devono esistere le cartelle dei nodi
    tree_weights_root="/home/mcaroti/EMB",

    leave_out_n_subjects=5,
)

test_kwargs = [
    dict(),
]
