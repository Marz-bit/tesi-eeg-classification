#versione ridotta della versione utilizzata per il training

#utilizzo label encoder per struttura gerarchica
is_tree = (classifier_type is ClassifierType.TREE_OF_EXPERTS_EEGNET)
is_binary_grouping = bool(getattr(label_encoder, "binary_grouping", False))

if is_tree or (not is_binary_grouping):
    # MULTICLASS (qualsiasi K>=2) oppure Tree: non filtrare
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
else:
    # BINARY GROUPING: filtra e rimappa a {0,1}
    y_train, train_mask = label_encoder.transform(y_train, return_mask=True)
    x_train = x_train[train_mask]
    ep_group = ep_group[train_mask]

    y_test, test_mask = label_encoder.transform(y_test, return_mask=True)
    x_test = x_test[test_mask]

# bilanciamento con pesi di classe inversamente proporzionali alla frequenza → classi rare pesano di più nella loss
if weight_classes and not balance_train_data:
    # class_weight = {i: 1 - (np.sum(y_train == i) / y_train.size) for i in range(len(label_encoder.classes_))}
    # class weights "balanced" (sklearn-like): w_i = N / (K * n_i)
    n_classes = len(label_encoder.classes_)
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0  # sicurezza (evita divisione per zero)
    N = y_train.size
    class_weight = {i: float(N / (n_classes * counts[i])) for i in range(n_classes)}
else:
    class_weight = None

# Mescola l’ordine delle finestre di training, per non avere blocchi di epoche consecutive tutte della stessa classe
if shuffle:
    ind = np.arange(len(y_train))
    np.random.shuffle(ind)
    x_train = x_train[ind]
    y_train = y_train[ind]
    ep_group = ep_group[ind]

# fa sì che quando istanzi il Tree, leaf_labels sia identico a label_encoder.classes_
# l’output è (B,9) e la colonna 0 corrisponde a leaf_labels[0]
if classifier_type is ClassifierType.TREE_OF_EXPERTS_EEGNET:
    classifier_kwargs.setdefault("leaf_labels", list(label_encoder.classes_))

model_kwargs = dict(classifier_kwargs)
# roba che NON deve mai entrare nel costruttore del modello
for k in ("weight_classes", "weight_file", "model_file", "train_model"):
    model_kwargs.pop(k, None)

clf = init_classifier(classifier_type, x_train[0].shape, len(label_encoder.classes_),
                      fs=fs, **model_kwargs)



##ALLENAMENTO + TL
if self._classifier_type is ClassifierType.TREE_OF_EXPERTS_EEGNET:
    test_subjects = [int(s) for s in np.unique(all_subj[test_ind])]

    # resume?
    if test_subjects == self._check_point.get("test_subj", []) and self._check_point.get("nn_cp_file",
                                                                                         None):
        self._classifier_kwargs["weight_file"] = self._check_point["nn_cp_file"]
        self._init_post_tl_test(test_subjects)
        continue

    if self._tree_weights_root is None:
        raise ValueError("tree_weights_root non impostato (serve per trovare i pesi binari per nodo).")

    # 1) init JSON (binari coerenti con lo split LPO)
    init_json = build_tree_weights_json(self._tree_weights_root, test_subjects)

    # 2) init Tree (leaf_labels coerenti con encoder)
    tree_kwargs = dict(classifier_kwargs)

    allowed = {"leaf_labels", "learning_rate", "eegnet_kwargs", "save_path", "node_loss_weight"} #"node_loss_weight"
    tree_kwargs = {k: v for k, v in tree_kwargs.items() if k in allowed}

    tree_kwargs.setdefault("leaf_labels", list(self._label_encoder.classes_))

    clf = init_classifier(
        self._classifier_type,
        db.get_data(train_ind[0]).shape,
        len(self._label_encoder.classes_),
        fs=db.get_fs(),
        **tree_kwargs
    )

    # 3) inizializza dai binari
    clf.load_weights(str(init_json)) #con lui commentato pesi random

    print("[TREE] Init from binary JSON:", init_json)

    print("[DEBUG TREE] train subjects:", np.unique(all_subj[train_ind]))
    print("[DEBUG TREE] test subjects :", test_subjects)

    # 4) TRAIN end-to-end sul TRAIN del LPO split (stessa logica dei binari)
    _fit_clf(
        clf, db, y, train_ind, db.get_epoch_group(),
        shuffle=self._shuffle,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=batch_size,
        give_up=give_up,
        patience=patience,
        weight_classes=not self._balance_data,
        verbose=verbose,
    )

    # 5) salva pesi TREE per quello split
    nn_file = Path(clf.save_weights())
    lpo_tag = "testsubj-" + "-".join(map(str, test_subjects))
    unique_file = nn_file.with_name(f"TREE_{lpo_tag}.weights.h5")
    k = 1
    while unique_file.exists():
        unique_file = nn_file.with_name(f"TREE_{lpo_tag}_{k}.weights.h5")
        k += 1
    unique_file.parent.mkdir(parents=True, exist_ok=True)
    #import shutil
    #shutil.move(str(nn_file), str(unique_file))

    # aggiorna per TL
    self._classifier_kwargs["weight_file"] = str(unique_file)
    self._check_point["nn_cp_file"] = str(unique_file)
    self._check_point["test_subj"] = test_subjects

    del clf

    # 6) ora fai evaluation+TL per i soggetti left-out come già fai
    self._init_post_tl_test(test_subjects)
    continue

