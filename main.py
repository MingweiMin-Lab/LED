
if __name__ == '__main__':
    from train import train_model as tm
    from predictor import predictor as pr
    from cell_tracking import tracker as ct
    import datetime

    now = datetime.datetime.now()
    print('START!', now)

    tm()
    now = datetime.datetime.now()
    print('train DONE!', now)

    pr()
    now = datetime.datetime.now()
    print('predict DONE!', now)

    ct()
    now = datetime.datetime.now()
    print('track DONE!', now)


    import pandas as pd
    import numpy as np
    from convert2CTCFormat import lineage_mask

    mask_dir = r"D:\Haishan\track\data\CTC_dataset\hela1\data\mask"   ###  SIM+=GT
    track_dir = r"D:\Haishan\track\data\CTC_dataset\hela1\result\2026-05-08track_results.csv"
    centroid = r"D:\Haishan\track\data\CTC_dataset\hela1\result\2026-05-08-centroid.npy"

    tracks = pd.read_csv(track_dir).to_numpy()
    cnt = np.load(centroid, allow_pickle=True)
    lineage_mask(mask_dir, tracks, cnt, save_path=r"D:\Haishan\track\data\CTC_dataset\hela1\result", res_path='RES1')
    lineage_mask(mask_dir, tracks, cnt, save_path=r"D:\Haishan\track\data\CTC_dataset\hela1\result", res_path='RES1', saveRGB=True)

    # # # # Import the required packages
    import numpy as np
    from ctcmetrics.ctc_metrics.scripts.evaluate import evaluate_sequence  #, validate_sequence
    from multiprocessing import cpu_count
    # # # Validate the sequence
    # valid = validate_sequence(r"I:\CellTrack\dataset\Fluo-N2DL-HELA\RES1")

    # Evaluate the sequence
    res = evaluate_sequence(r"D:\Haishan\track\data\CTC_dataset\hela1\result\RES1",
                            r'G:\CellTrack\dataset\Fluo-N2DL-HeLa\01_GT',
                            threads=1)

    print(res)
    try:
        bio = np.mean((res["CT"], res["TF"], res["CCA"], res["BC(0)"]))
        clb = np.mean((res["LNK"], bio))
        print('OP_clb:', clb, "LNK:", res["LNK"], "BIO:", bio,
              "CT:", res["CT"], "TF:", res["TF"],
              "CCA:", res["CCA"], "BC(0):", res["BC(0)"])
    except:
        print("LNK:", res["LNK"], "CT:", res["CT"], "TF:", res["TF"],
              "CCA:", res["CCA"], "BC(0):", res["BC(0)"])