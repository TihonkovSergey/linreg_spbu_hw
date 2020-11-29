def mse(y_true, y_pred):
    assert len(y_true) == len(y_pred),\
        f"Truth and predicted Y sizes must be equal, but {len(y_true)} != {len(y_pred)}"
    value = 0
    for i in range(len(y_true)):
        value += (y_true[i] - y_pred[i]) ** 2
    value /= len(y_true)
    return value


def r2(y_true, y_pred):
    assert len(y_true) == len(y_pred),\
        f"Truth and predicted Y sizes must be equal, but {len(y_true)} != {len(y_pred)}"

    # residual sum of squares:
    rss = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))

    # mean value
    mean_y = sum(y_true) / len(y_true)

    # total sum of squares
    tss = sum((y_true[i] - mean_y) ** 2 for i in range(len(y_true)))

    # coefficient of determination
    r_2 = 1 - rss / tss
    return r_2
