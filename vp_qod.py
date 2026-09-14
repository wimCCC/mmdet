def my_equation_add_harmonic_iou_loss(reg_loss, conf_values, pos_conf_values, ious, pos_mask):
    beta, alpha, gamma = 1.2, 1.5, 0.8
    iou_loss = alpha * (1 - ious) * (1 + ious) ** gamma
    reg_loss = reg_loss + iou_loss
    return reg_loss

def my_equation_add_harmonic_conf_loss(cls_loss, conf_values, pos_conf_values, pos_gtconf_values, pos_mask):
    beta, alpha, gamma = 1.2, 1.5, 0.8
    conf_loss = alpha * (1 - pos_gtconf_values) * (1 + pos_gtconf_values) ** gamma
    cls_loss = cls_loss + conf_loss * pos_mask.float()
    return cls_loss

def HQOD_loss(pos_reg_loss, pos_cls_loss, conf_values, pos_gtconf_values, pos_conf_values, matched_iou_vals, cls_branch_factor, reg_branch_factor, mask):
    pos_reg_loss = my_equation_add_harmonic_iou_loss(pos_reg_loss, None, None, matched_iou_vals, mask)
    pos_reg_loss = my_equation_add_harmonic_conf_loss(pos_reg_loss, conf_values, pos_conf_values, pos_gtconf_values, mask)
    return pos_reg_loss, pos_cls_loss
