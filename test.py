from bionic_apps.ai import ClassifierType
from bionic_apps.bci_train_test_suites import WithinExpHandler, CrossSubjectExpHandler
from bionic_apps.databases import Databases
from bionic_apps.databases.eeg.defaults import BOTH_LEGS, REST, RIGHT_HAND, SPATIAL, ROTATION, WORD, SUBTRACT, MUSIC, \
    LEFT_HAND
from bionic_apps.feature_extraction import FeatureType
from bionic_apps.preprocess.io import SubjectHandle

LOG_DIR = 'bci_tests/eeg_net'
TEST_NAME = 'EMB_SOFT'

# HPC params:
partition = 'gpu'  # ['cpu', 'cpu_lowpriority', 'gpu', 'gpu_long', 'gpu_lowpriority']
exclude = 'wald'
gpu_type = 1  # [1, 2, 3]
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
    do_artefact_rejection=False, #CAMBIATO DA TRUE
    balance_data=True,
    subject_handle=SubjectHandle.INDEPENDENT_DAYS,
    n_splits=5,
    classifier_type=ClassifierType.EEG_NET,
    classifier_kwargs=dict(
        validation_split=.2,
        epochs=150,
        patience=20,
        give_up=40,
        weight_classes=True,
        #learning_rate=.0001
    ),
    log_file='out.csv', base_dir='.',
    #selected_labels=[BOTH_LEGS, LEFT_HAND, MUSIC, REST, RIGHT_HAND, ROTATION, SPATIAL, SUBTRACT, WORD], #commentalo
    save_classifiers=True,
    subjects='all',
    tl_subjects='all',
    binarize_labels=True,  #per flat e 3 classi False
    #first_set={MUSIC, SUBTRACT, WORD},
    #second_set={RIGHT_HAND, LEFT_HAND, ROTATION, BOTH_LEGS, REST, SPATIAL}
)

# generating test params here...
test_kwargs = [
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       first_set={MUSIC, WORD, SUBTRACT},
    #       second_set={RIGHT_HAND, LEFT_HAND, ROTATION, BOTH_LEGS, REST, SPATIAL}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[MUSIC, WORD],
    #       first_set={MUSIC},
    #       second_set={WORD}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[MUSIC, SUBTRACT, WORD],
    #       first_set={SUBTRACT},
    #       second_set={MUSIC, WORD}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[REST, SPATIAL, ROTATION, RIGHT_HAND, LEFT_HAND, BOTH_LEGS],
    #       first_set={REST, SPATIAL, ROTATION},
    #       second_set={RIGHT_HAND, LEFT_HAND, BOTH_LEGS}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[LEFT_HAND, RIGHT_HAND],
    #       first_set={LEFT_HAND},
    #       second_set={RIGHT_HAND}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[REST, SPATIAL, ROTATION],
    #      first_set={REST},
    #      second_set={SPATIAL, ROTATION}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[ROTATION, SPATIAL],
    #       first_set={ROTATION},
    #       second_set={SPATIAL}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[LEFT_HAND, RIGHT_HAND, BOTH_LEGS],
    #      first_set={BOTH_LEGS},
    #      second_set={RIGHT_HAND, LEFT_HAND}),

    # #EEG 9 Classi- 6 classi
    # dict(
    #     db_name=Databases.CYBATHLON2028_INIT,
    #     leave_out_n_subjects=5,
    #     selected_labels=[BOTH_LEGS, MUSIC, REST, ROTATION, SUBTRACT, WORD],
    # )

    # #INVERTED N5, N6, N7. N8
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[BOTH_LEGS, RIGHT_HAND],
    #      first_set={BOTH_LEGS},
    #      second_set={RIGHT_HAND}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[REST, ROTATION],
    #      first_set={REST},
    #      second_set={ROTATION}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[REST, SPATIAL, ROTATION],
    #      first_set={SPATIAL},
    #      second_set={REST, ROTATION}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[LEFT_HAND, RIGHT_HAND, BOTH_LEGS],
    #      first_set={LEFT_HAND},
    #      second_set={RIGHT_HAND, BOTH_LEGS}),

    # #3-CLASS LAST LEAFS
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[REST, SPATIAL, ROTATION],
    #      ),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[LEFT_HAND, RIGHT_HAND, BOTH_LEGS],
    #      ),

    # #6 CLASSES
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #      selected_labels=[MUSIC, WORD, SUBTRACT, ROTATION, BOTH_LEGS, REST],
    #       first_set={MUSIC, WORD, SUBTRACT},
    #       second_set={ROTATION, BOTH_LEGS, REST}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[SUBTRACT, WORD],
    #       first_set={SUBTRACT},
    #       second_set={WORD}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[MUSIC, SUBTRACT, WORD],
    #       first_set={MUSIC},
    #       second_set={SUBTRACT, WORD}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[REST, BOTH_LEGS, ROTATION],
    #      first_set={ROTATION},
    #      second_set={BOTH_LEGS, REST}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[BOTH_LEGS, REST],
    #       first_set={BOTH_LEGS},
    #       second_set={REST}),

    # # 2 VERSIONE 6 CLASSI BASATA SU 9 CLASSI
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[REST, BOTH_LEGS, ROTATION],
    #      first_set={REST, ROTATION},
    #      second_set={BOTH_LEGS}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[ROTATION, REST],
    #      first_set={ROTATION},
    #      second_set={REST}),

    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       first_set={BOTH_LEGS, REST},
    #       second_set={MUSIC, WORD, SUBTRACT, RIGHT_HAND, LEFT_HAND, ROTATION , SPATIAL}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[SPATIAL, WORD],
    #       first_set={SPATIAL},
    #       second_set={WORD}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[MUSIC, SPATIAL, WORD],
    #       first_set={MUSIC},
    #       second_set={SPATIAL, WORD}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[RIGHT_HAND, LEFT_HAND, ROTATION, SPATIAL, SUBTRACT, MUSIC, WORD],
    #       first_set={RIGHT_HAND, LEFT_HAND, ROTATION},
    #       second_set={ MUSIC, WORD, SPATIAL, SUBTRACT}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[LEFT_HAND, ROTATION],
    #       first_set={LEFT_HAND},
    #       second_set={ROTATION}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[REST, BOTH_LEGS],
    #      first_set={REST},
    #      second_set={BOTH_LEGS}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #       leave_out_n_subjects=5,
    #       selected_labels=[MUSIC, SPATIAL, WORD, SUBTRACT],
    #       first_set={MUSIC, SPATIAL, WORD},
    #       second_set={SUBTRACT}),
    # dict(db_name=Databases.CYBATHLON2028_INIT,
    #      leave_out_n_subjects=5,
    #      selected_labels=[LEFT_HAND, RIGHT_HAND, ROTATION],
    #      first_set={RIGHT_HAND},
    #      second_set={ROTATION, LEFT_HAND}),

    dict(db_name=Databases.CYBATHLON2028_INIT,
          leave_out_n_subjects=5,
          selected_labels=[MUSIC, SUBTRACT, WORD],
          first_set={MUSIC},
          second_set={SUBTRACT, WORD}),
    dict(db_name=Databases.CYBATHLON2028_INIT,
          leave_out_n_subjects=5,
          selected_labels=[WORD, SUBTRACT],
          first_set={SUBTRACT},
          second_set={WORD}),
    dict(db_name=Databases.CYBATHLON2028_INIT,
         leave_out_n_subjects=5,
         selected_labels=[REST, ROTATION],
         first_set={REST},
         second_set={ROTATION}),
    dict(db_name=Databases.CYBATHLON2028_INIT,
         leave_out_n_subjects=5,
         selected_labels=[REST, SPATIAL, ROTATION],
         first_set={SPATIAL},
         second_set={REST, ROTATION}),

]
