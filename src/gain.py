# -*- coding: utf-8 -*-

import numpy as np


def channel_gain(K, N, C, xyUEs, xySBSs, xi):
    """
    :param xyUEs: The coordinates of the users. A matrix of size Kx2.
        xyUEs[k, :] is the coordinate of UE k. xy_UEs[k, 0] is the abscissa of
        UE k and xy_UEs[k, 1] is the ordinate of UE k.
    :param xySBSs: The coordinates of the SBSs. A matrix of size Nx2.
        xySBSs[n, :] is the coordinate of SBS n. xy_SBSs[n, 0] is the abscissa of
        SBS n and xySBSs[n, 1] is the ordinate of SBS n.
    :param xyMBS: The coordiante of the MBS. A vector of size 1x2.
        xyMBS[0] is the abscissa of the MBS and xy_MBS[1] is the ordinate of the MBS.
    :param ξ₁: Shadowing effect for the MBS
    :param ξ₂: Shadowing effect for the SBSs
    :return: The path-loss between the users and the BSs. PL_SBSs is a matrix
        of size KxN (taken from 3GPP reference for UMi).
        PL_MBS is a vector of size Kx1 (taken from 3GPP reference for UMa).
        PL: is a matrix of size Kx(N+1) that includes the path loss between users and BSs.
    """
    # plMBS = zeros(K);  # Path-loss between users and the MBS
    # if !isempty(xyMBS)
    #     for k in 1:K
    #         plMBS[k] = 10 ^ (-1.53 - 3.76 * log10(norm(xyUEs[:, k] - xyMBS)) + ξ₁ / 10.0);
    #     end
    # end

    plSBSs = np.zeros((K, N))  # Path-loss between users and the SBSs
    for k in range(K):
        for n in range(N):
            # Shadowing
            # Xi = np.random.normal(0.0, 8.0)
            plSBSs[k, n] = 10 ** (-3.06 - 3.67 * np.log10(np.linalg.norm(xyUEs[:, k] - xySBSs[:, n])) + xi / 10.0)
    h = np.random.exponential(scale=1.0, size=(C, K, N))
    g = np.multiply(h, plSBSs)
    # pl = hcat(plSBSs, plMBS);
    return g
