"""Copy-paste of rare small objects: bank, placement rules, labels."""

import numpy as np

from aerotrack.data.copy_paste import Bank, paste, rare_classes


def test_rare_classes_by_share():
    assert rare_classes({0: 80, 1: 15, 2: 4, 3: 1}, 0.05) == [2, 3]


def test_paste_stays_on_free_background():
    rng = np.random.default_rng(0)
    img = np.zeros((200, 300, 3), np.uint8)
    boxes = np.array([[0, 0, 150, 200]], float)       # left half is labelled
    ignore = np.array([[150, 0, 300, 100]], float)    # top right is ignored
    bank = Bank([np.full((10, 12, 3), 200, np.uint8), np.full((8, 8, 3), 100, np.uint8)],
                np.array([7, 6]))
    out, new, cls = paste(img.copy(), boxes, ignore, bank, scale=1.0, n=6, rng=rng)
    assert len(new) == len(cls) > 0 and set(cls) <= {6, 7}
    assert (new[:, 0] >= 150).all() and (new[:, 1] >= 100).all()  # only the free quarter
    assert (new[:, 2] <= 300).all() and (new[:, 3] <= 200).all()
    for a in range(len(new)):                       # pasted objects do not overlap each other
        for b in range(a + 1, len(new)):
            x = min(new[a, 2], new[b, 2]) - max(new[a, 0], new[b, 0])
            y = min(new[a, 3], new[b, 3]) - max(new[a, 1], new[b, 1])
            assert x <= 0 or y <= 0
    x0, y0, x1, y1 = new[0].astype(int)
    assert out[y0:y1, x0:x1].any() and not out[:, :150].any()   # pixels written, only there


def test_paste_scales_crops_with_the_receiving_image():
    bank = Bank([np.full((20, 40, 3), 255, np.uint8)], np.array([2]))
    _, new, _ = paste(np.zeros((300, 300, 3), np.uint8), np.zeros((0, 4)), np.zeros((0, 4)),
                      bank, scale=0.5, n=1, rng=np.random.default_rng(1))
    assert (new[0, 2] - new[0, 0], new[0, 3] - new[0, 1]) == (20, 10)
