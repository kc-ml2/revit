from escnn import gspaces


# ESCNN subgroup IDs inside the repository's D4, and matching square-grid
# action IDs (rotate 0..3 quarter turns, with a left-right flip first for 4..7).
D4_SUBGROUPS = {
    'D4': ((0, 4), tuple(range(8))),
    'C4': ((None, 4), (0, 1, 2, 3)),
    'D2_4': ((0, 2), (0, 2, 4, 6)),
    'C2': ((None, 2), (0, 2)),
    'Cs': ((0, 1), (0, 4)),
    'C1': ((None, 1), (0,)),
}
GROUP_ALIASES = {'D2': 'Cs', 'Z2': 'C1'}
SQUARE_GROUPS = (*D4_SUBGROUPS, *GROUP_ALIASES)


def canonical_square_group(group):
    return GROUP_ALIASES.get(group, group)


def grid_actions(group):
    return D4_SUBGROUPS[canonical_square_group(group)][1]


def get_gspace(group_str):
    group = canonical_square_group(group_str)
    if group == 'C1':
        return gspaces.trivialOnR2()
    elif group == 'Cs':
        return gspaces.flip2dOnR2()
    elif group == 'C2':
        return gspaces.rot2dOnR2(N=2)
    elif group == 'D2_4':
        return gspaces.flipRot2dOnR2(N=2)
    elif group == 'C4':
        return gspaces.rot2dOnR2(N=4)
    elif group_str == 'C8':
        return gspaces.rot2dOnR2(N=8)
    elif group_str == 'C12':
        return gspaces.rot2dOnR2(N=12)
    elif group_str == 'D4':
        return gspaces.flipRot2dOnR2(N=4)
    elif group_str == 'D8':
        return gspaces.flipRot2dOnR2(N=8)
    elif group_str == 'D12':
        return gspaces.flipRot2dOnR2(N=12)
    elif group_str == 'C16':
        return gspaces.rot2dOnR2(N=16)
    elif group_str == 'D16':
        return gspaces.flipRot2dOnR2(N=16)
    else:
        raise ValueError(f"Invalid group string: {group_str}")
