def polcovshow(img, l=2.0, lex=False, imgout=False):
    """Displays a PolSAR covariance"""

    import numpy as np
    from matplotlib.pyplot import imshow, show, gcf, gca, axis

    dimg = np.zeros([img.shape[0], img.shape[1],3])
    dimg = dimg - dimg.min()
    if not lex:
      dimg[:,:,0] = np.sqrt(abs(img[:,:,1,1]) / np.max(np.sqrt(abs(img[:,:,1,1]))))
      dimg[:,:,1] = np.sqrt(abs(img[:,:,2,2]) / np.max(np.sqrt(abs(img[:,:,2,2]))))
      dimg[:,:,2] = np.sqrt(abs(img[:,:,0,0]) / np.max(np.sqrt(abs(img[:,:,0,0]))))
    else:
      dimg[:,:,0] = np.sqrt(abs(img[:,:,0,0]) / np.max(np.sqrt(abs(img[:,:,0,0]))))
      dimg[:,:,1] = np.sqrt(abs(img[:,:,1,1]) / np.max(np.sqrt(abs(img[:,:,1,1]))))
      dimg[:,:,2] = np.sqrt(abs(img[:,:,2,2]) / np.max(np.sqrt(abs(img[:,:,2,2]))))
    m0 = dimg[:,:,0].mean()
    m1 = dimg[:,:,1].mean()
    m2 = dimg[:,:,2].mean()
    dimg[:,:,0] = np.clip(dimg[:,:,0], 0, l*m0)
    dimg[:,:,1] = np.clip(dimg[:,:,1], 0, l*m1)
    dimg[:,:,2] = np.clip(dimg[:,:,2], 0, l*m2)
    dimg[:,:,0] = dimg[:,:,0] / np.max(dimg[:,:,0])
    dimg[:,:,1] = dimg[:,:,1] / np.max(dimg[:,:,1])
    dimg[:,:,2] = dimg[:,:,2] / np.max(dimg[:,:,2])

    imshow(dimg, interpolation='None')
    axis('off')
    gcf().tight_layout(pad=0)

    if imgout:
        return dimg
