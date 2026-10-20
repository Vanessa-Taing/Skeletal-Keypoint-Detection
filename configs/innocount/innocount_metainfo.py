dataset_info = dict(
    dataset_name='innocount_steel_profiles',
    paper_info=dict(
        author='InnoCount project',
        title='Generic Visual Keypoint Detection for Steel Structural Profiles',
        container='Project dataset',
        year='2026',
        homepage='',
    ),
    keypoint_info={
        0: dict(name='G1', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='G2', id=1, color=[0, 255, 0], type='', swap=''),
        2: dict(name='G3', id=2, color=[0, 0, 255], type='', swap=''),
        3: dict(name='G4', id=3, color=[255, 255, 0], type='', swap=''),
        4: dict(name='G5', id=4, color=[255, 0, 255], type='', swap=''),
        5: dict(name='G6', id=5, color=[0, 255, 255], type='', swap=''),
    },
    skeleton_info={
        0: dict(link=('G1', 'G2'), id=0, color=[255, 0, 0]),
        1: dict(link=('G2', 'G3'), id=1, color=[0, 255, 0]),
        2: dict(link=('G3', 'G4'), id=2, color=[0, 0, 255]),
        3: dict(link=('G4', 'G5'), id=3, color=[255, 255, 0]),
        4: dict(link=('G5', 'G6'), id=4, color=[255, 0, 255]),
    },
    joint_weights=[1., 1., 1., 1., 1., 1.],
    sigmas=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
)
