"""
08. Extremely Randomized Trees (Geurts, Ernst & Wehenkel, 2006)
===============================================================

Extra-Trees push the Random Forest idea one notch further. A Random Forest
(script 02) randomizes *which features* each split may consider, but still
searches for the *best threshold* on those features. Extra-Trees also randomize
the threshold: for each candidate feature it draws a *random* cut point and
keeps the best of those random cuts. It also trains on the whole dataset (no
bootstrap by default).

Architecture Diagram / Layout:
    Random Forest split :  pick m random features -> search BEST threshold on each
    Extra-Trees split   :  pick m random features -> draw ONE RANDOM threshold each
                                                   -> keep the best random cut
    -> N such fully-grown, extra-random trees -> average their votes

Key insights / educational takeaways:
    * Turning the bias/variance dial further than Random Forest: extra randomness
      raises each tree's bias slightly but decorrelates trees more, cutting
      ensemble variance — often matching RF accuracy at lower cost.
    * Random split thresholds mean no threshold search, so training is faster.
    * Compare the feature-importance chart directly against script 02
      (random-forest): the rankings are usually similar but smoothed out.

Run:
    python "08.extra-trees.py" --n-estimators 200 --dataset covtype
    python "08.extra-trees.py" --limit 2000        # fast smoke test
"""

import os
from sklearn.ensemble import ExtraTreesClassifier
import tabular_common as mc


def main():
    p = mc.build_argparser("Extra-Trees Tabular Classifier", max_depth=None,
                           n_estimators=200)
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    print(f"Training Extra-Trees (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth})...")
    model = ExtraTreesClassifier(n_estimators=args.n_estimators,
                                 max_depth=args.max_depth,
                                 random_state=args.seed, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    mc.report_classification(y_test, preds, proba, class_names=class_names,
                             model_name="Extra-Trees",
                             save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        mc.plot_feature_importances(
            model.feature_importances_, feature_names,
            os.path.join(_here(), "et_feature_importances.png"),
            "Extra-Trees Feature Importances")


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
