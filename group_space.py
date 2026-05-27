from escnn import gspaces

def get_gspace(group_str):
    if group_str == 'C4':
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
    elif group_str == 'D2':
        return gspaces.flip2dOnR2()
    elif group_str == 'Z2':
        return gspaces.trivialOnR2()
    else:
        raise ValueError(f"Invalid group string: {group_str}")