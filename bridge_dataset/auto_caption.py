import numpy as np

DIRECTIONAL_CAPTIONS = [
    ["move farther", "move closer"],
    ["move left", "move right"],
    ["move up", "move down"],
    ["", ""],
    ["", ""],
    ["rotate left", "rotate right"],
    ["open gripper", "close gripper"],
]

# hardcode for now
action_mean = np.array(
    [
        0.00021160562755540013,
        0.00012613687431439757,
        -0.0001702189474599436,
        -0.00015061876911204308,
        -0.00023830759164411575,
        0.00025645774439908564,
        0.58806312084198,
    ]
)
action_std = np.array(
    [
        0.009637202136218548,
        0.013506611809134483,
        0.01251861359924078,
        0.02806786075234413,
        0.030169039964675903,
        0.07632622122764587,
        0.4883806109428406,
    ]
)


def auto_caption(traj, time_delta=5):
    """
    Automatically caption a trajectory using proprioceptive deltas.
        traj: a trajectory from the dataset
        metadata: dict containing the mean and std of the actions
        time_delta: the time difference between proprioceptive states
    Possible captions, by dimension (positive direction, negative direction):
        0: move farther, move closer
        1: move left, move right
        2: move up, move down
        3: (not used)
        4: (not used)
        5: rotate left, rotate right (counter-clockwise, clockwise)
        6: open gripper, close gripper
    We concatenate the two captions with the highest z-score.
    """

    proprios = [obs["state"] for obs in traj["observations"]]
    t_proprio = proprios
    tplus_proprio = proprios[time_delta:]
    # extend the last proprio to match the length of the others
    # print(tplus_proprio.shape, )
    tplus_proprio = np.concatenate(
        [tplus_proprio]
        + [tplus_proprio[-1:] for _ in range(len(t_proprio) - len(tplus_proprio))],
        axis=0,
    )
    proprio_deltas = tplus_proprio - t_proprio

    z_scores = (proprio_deltas / time_delta) / action_std[None]
    z_signs = np.sign(z_scores)
    z_scores = np.abs(z_scores)

    z_scores[:, (3, 4, 6)] = 0  # zero out unused dimensions
    top_z_idxs = np.argsort(z_scores, axis=1)
    top_z_idxs = top_z_idxs[:, -2:]

    captions = []
    for i in range(len(z_scores)):
        caption = ""
        for dim_idx in top_z_idxs[i, ::-1]:
            # only caption if z-score is high enough
            if z_scores[i, dim_idx] < 0.5:
                continue
            caption += DIRECTIONAL_CAPTIONS[dim_idx][z_signs[i, dim_idx] < 0] + "; "

        # treat gripper separately
        if proprio_deltas[i, 6] > 0.3:
            caption += "open gripper; "
        elif proprio_deltas[i, 6] < -0.3:
            caption += "close gripper; "

        captions.append(caption)

    return captions
