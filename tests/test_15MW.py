import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from MITRotor import BEM, IEA15MW, BEMGeometry, NoTipLoss, PrandtlTipLoss, ConstantInduction, ClassicalMomentum, UnifiedMomentum, MadsenMomentum, NoTangentialInduction, DefaultTangentialInduction, BEMSolution

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


def plot_radial_distributions(sol: BEMSolution, save_to: Path):
    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].set_ylabel('$C_P$ [-]')
    axes[1].set_ylabel('$C_T$ [-]')
    axes[2].set_ylabel('$a_n$ [-]')

    # Constant induction model

    axes[0].plot(sol.geom.mu, sol.Cp(grid="annulus"), label="[JYL]: avg: 0.476")
    axes[1].plot(sol.geom.mu, sol.Ct(grid="annulus"), label="[JYL]: avg: 0.781")
    axes[2].plot(sol.geom.mu, sol.a(grid="annulus"), label="[JYL]: avg: 0.333")

    # r_sam  = np.loadtxt('/Users/stormmata/Downloads/debugging/r.csv', delimiter=',')
    # cp_sam = np.loadtxt('/Users/stormmata/Downloads/debugging/cp.csv', delimiter=',')
    # ct_sam = np.loadtxt('/Users/stormmata/Downloads/debugging/ct.csv', delimiter=',')
    # a_sam  = np.loadtxt('/Users/stormmata/Downloads/debugging/a.csv', delimiter=',')

    # axes[0].plot(r_sam, cp_sam, label="[SAM]: avg: 0.471; -1.05%")
    # axes[1].plot(r_sam, ct_sam, label="[SAM]: avg: 0.776; -0.64%")
    # axes[2].plot(r_sam, a_sam, label="[SAM]: avg: 0.333; -0.00%")

    # axes[0].set_title('Constant Momentum Model')

    # axes[2].set_ylim([0,0.5])

    # [ax.legend(loc="lower center") for ax in axes]

    axes[-1].set_xlabel("$r/R$ [-]")
    plt.xlim(0, 1)
    plt.savefig(save_to, dpi=500, bbox_inches="tight")


def plot_azimuthal_variations(sol: BEMSolution, save_to: Path):
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=np.array([6, 6]))

    azim = np.rad2deg(sol.geom.theta)
    for mu, ax in zip([0.9, 0.6, 0.3], axes):
        # find closest grid point to target radial position.
        idx = np.searchsorted(sol.geom.mu, mu)
        ax.plot(azim, sol.W(grid="sector")[idx, :])

    plt.xlim(0, 360)

    plt.savefig(save_to, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # Initialize rotor with increased radial resolution.
    rotor = IEA15MW()

    bem = BEM(rotor=rotor, geometry=BEMGeometry(Nr=50, Ntheta=160), tiploss_model=PrandtlTipLoss(), momentum_model=MadsenMomentum(), tangential_induction_model=DefaultTangentialInduction())

    wrf_roverR = (np.array([0.041,0.057,0.072,0.088,0.104,0.120,0.136,0.152,0.168,0.183,0.199,0.215,0.231,0.247,0.263,0.279,0.294,0.310,0.326,0.342,0.358,0.374,0.390,0.405,0.421,0.437,0.453,0.469,0.485,0.501,0.516,0.532,0.548,0.564,0.580,0.596,0.612,0.627,0.643,0.659,0.675,0.691,0.707,0.723,0.738,0.754,0.770,0.786,0.802,0.818,0.834,0.849,0.865,0.881,0.897,0.913,0.929,0.945,0.960,0.976,0.992]) * 117) / (242/2)
    wrf_AoA    = [54.408,47.903,42.057,36.954,32.537,28.731,25.486,22.700,20.301,18.256,16.420,14.885,13.538,12.446,11.567,10.862,10.310,9.855,9.500,9.181,8.928,8.675,8.466,8.270,8.100,7.942,7.802,7.678,7.564,7.464,7.368,7.287,7.201,7.128,7.059,7.000,6.949,6.910,6.882,6.879,6.891,6.950,7.022,7.117,7.213,7.297,7.362,7.398,7.378,7.317,7.230,7.142,7.034,6.935,6.825,6.727,6.621,6.553,6.525,6.507,6.483]
    wrf_cl     = [0.000,0.000,0.000,1.188,1.304,1.414,1.538,1.598,1.657,1.725,1.836,1.881,1.885,1.880,1.849,1.809,1.741,1.694,1.656,1.621,1.593,1.493,1.468,1.446,1.426,1.407,1.390,1.336,1.322,1.310,1.299,1.289,1.279,1.271,1.237,1.230,1.224,1.220,1.217,1.216,1.218,1.225,1.211,1.222,1.232,1.241,1.248,1.252,1.250,1.243,1.234,1.224,1.213,1.202,1.189,1.178,1.166,1.159,1.156,1.154,1.151]
    wrf_cd     = [0.350,0.350,0.350,0.441,0.352,0.278,0.222,0.185,0.157,0.131,0.077,0.057,0.043,0.033,0.027,0.024,0.019,0.018,0.018,0.017,0.017,0.015,0.014,0.014,0.014,0.014,0.014,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.011,0.011,0.011,0.011,0.011,0.011,0.011,0.011,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.010,0.009,0.009,0.009,0.009,0.009,0.009]
    wrf_phi    = [70.294,63.370,57.039,51.297,46.172,41.578,37.522,33.913,30.739,27.931,25.466,23.319,21.447,19.837,18.452,17.258,16.233,15.338,14.559,13.864,13.244,12.679,12.164,11.687,11.246,10.836,10.455,10.099,9.766,9.453,9.158,8.881,8.620,8.375,8.143,7.923,7.712,7.510,7.316,7.128,6.948,6.773,6.606,6.445,6.292,6.148,6.014,5.890,5.779,5.678,5.590,5.514,5.454,5.411,5.392,5.400,5.445,5.519,5.619,5.728,5.833]

    ws        = np.array([6, 7, 8, 9, 10, 11, 12, 13])
    pitch_wrf = [-0.33797, -0.63879, -0.97996, -1.32680, 0.73192, 2.4859, 5.09471, 7.26585]
    tsr_wrf   = np.array([3.69324, 4.30878, 4.92432, 5.53986, 6.15540, 6.77094, 7.06110, 7.06110]) * 2*np.pi / 60 * 242/2 / ws

    mit_cp = np.zeros_like(pitch_wrf)
    mit_ct = np.zeros_like(pitch_wrf)

    # for i in range(len(pitch_wrf)):
    for i in range(1):


        # solve BEM for a control set point.
        pitch, tsr, yaw = np.deg2rad(pitch_wrf[i]), tsr_wrf[i], np.deg2rad(0.0)
        sol = bem(pitch, tsr, yaw, U=1.0)

        # pitch, tsr, yaw = np.deg2rad(0.522), 8.882, np.deg2rad(0.0)
        # sol = bem(pitch, tsr, yaw, U=1.0)

        # Plot
        # plot_radial_distributions(sol, figdir / "ConstantInduction_15MW_test.png")
        mit_cp[i] = sol.Cp()
        mit_ct[i] = sol.Ct()

    np.save("15MW_MadsenMomentum_NoLoss_TanInd.npy", {"cp": mit_cp, "ct": mit_ct})
    print("Attributes and methods in sol:")
    print(dir(sol))

    # print("sol attributes and values:")
    # for key, value in sol.__dict__.items():
    #     print(f"{key}: {value}")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5), constrained_layout=True, sharex=True)

    ax[0].plot(ws,mit_cp,linestyle='solid',linewidth=2, label='Reference', color='grey')
    ax[1].plot(ws,mit_ct,linestyle='solid',linewidth=2, label='WRF', color='#fc1703')
    # ax[0].plot(ws,ConstantInduction['cp'],linestyle='--',linewidth=2, label='MITRotor: Constant Induction', color='#1a6e03')
    # ax[0].plot(ws,UnifiedMomentum['cp'],linestyle='--',linewidth=2, label='MITRotor: Unified Momentum', color='#3b8a25')
    # ax[0].plot(ws,MadsenMomentum['cp'],linestyle='--',linewidth=2, label='MITRotor: Madsen Momentum', color='#6db858')

    # ax[0].set_ylabel('$C_P$ [-]')
    # ax[0].set_xlabel('Wind speed [m/s]')

    # ax[0].legend()

    # ax[1].plot(ref_ws,ref_ct,linestyle='solid',linewidth=2, label='Reference', color='grey')
    # ax[1].plot(wrf_ws,wrf_ct,linestyle='solid',linewidth=2, label='WRF', color='#fc1703')
    # ax[1].plot(ws,ConstantInduction['ct'],linestyle='--',linewidth=2, label='MITRotor: Constant Induction', color='#1a6e03')
    # ax[1].plot(ws,UnifiedMomentum['ct'],linestyle='--',linewidth=2, label='MITRotor: Unified Momentum', color='#3b8a25')
    # ax[1].plot(ws,MadsenMomentum['ct'],linestyle='--',linewidth=2, label='MITRotor: Madsen Momentum', color='#6db858')

    # ax[1].set_ylabel('$C_T$ [-]')
    # ax[1].set_xlabel('Wind speed [m/s]')

    # ax[1].legend()

    plt.savefig("15MW_wrf_mitrotor_ref_NoLoss_TanInd.png", bbox_inches="tight", dpi=600) 


    # ConstantInduction = np.load("ConstantInduction.npy", allow_pickle=True).item()
    # UnifiedMomentum   = np.load("UnifiedMomentum.npy", allow_pickle=True).item()
    # MadsenMomentum    = np.load("MadsenMomentum.npy", allow_pickle=True).item()

    # ConstantInduction = np.load("ConstantInduction_NoLoss_TanInd.npy", allow_pickle=True).item()
    # UnifiedMomentum   = np.load("UnifiedMomentum_NoLoss_TanInd.npy", allow_pickle=True).item()
    # MadsenMomentum    = np.load("MadsenMomentum_NoLoss_TanInd.npy", allow_pickle=True).item()

    # ConstantInduction = np.load("ConstantInduction_PrandTipLoss_NoTanInd.npy", allow_pickle=True).item()
    # UnifiedMomentum   = np.load("UnifiedMomentum_PrandTipLoss_NoTanInd.npy", allow_pickle=True).item()
    # MadsenMomentum    = np.load("MadsenMomentum_PrandTipLoss_NoTanInd.npy", allow_pickle=True).item()

    # ConstantInduction = np.load("ConstantInduction_PrandTipLoss_defaultTanInd.npy", allow_pickle=True).item()
    # UnifiedMomentum   = np.load("UnifiedMomentum_PrandTipLoss_defaultTanInd.npy", allow_pickle=True).item()
    # MadsenMomentum    = np.load("MadsenMomentum_PrandTipLoss_defaultTanInd.npy", allow_pickle=True).item()

    # ref_ws = np.array([3, 4, 5, 6, 7, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 15, 16, 18, 20, 25])
    # ref_cp = np.array([0.074, 0.364, 0.453, 0.481, 0.483, 0.485, 0.487, 0.484, 0.478, 0.470, 0.421, 0.368, 0.324, 0.255, 0.204, 0.166, 0.137, 0.097, 0.071, 0.037])
    # ref_ct = np.array([0.915, 0.926, 0.921, 0.895, 0.885, 0.873, 0.827, 0.789, 0.754, 0.721, 0.591, 0.490, 0.418, 0.318, 0.251, 0.203, 0.167, 0.119, 0.088, 0.049])

    # wrf_ws = np.array([4.71901576,5.95639165,6.94329198,7.92880503,8.92250125,9.42389107,9.92531459,10.42651939,10.94378281,11.45340602,11.95905813])
    # wrf_cp = np.array([0.48319912,0.50761431,0.5032652,0.49948904,0.4995687,0.49706891,0.49088654,0.48205099,0.43240675,0.38094035,0.33783701])
    # wrf_ct = np.array([0.93964934,0.88806915,0.88529706,0.8830269,0.83763999,0.79604799,0.75682813,0.72007519,0.58711934,0.49124122,0.42237043])

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5), constrained_layout=True, sharex=True)

    # ax[0].plot(ref_ws,ref_cp,linestyle='solid',linewidth=2, label='Reference', color='grey')
    # ax[0].plot(wrf_ws,wrf_cp,linestyle='solid',linewidth=2, label='WRF', color='#fc1703')
    # ax[0].plot(ws,ConstantInduction['cp'],linestyle='--',linewidth=2, label='MITRotor: Constant Induction', color='#1a6e03')
    # ax[0].plot(ws,UnifiedMomentum['cp'],linestyle='--',linewidth=2, label='MITRotor: Unified Momentum', color='#3b8a25')
    # ax[0].plot(ws,MadsenMomentum['cp'],linestyle='--',linewidth=2, label='MITRotor: Madsen Momentum', color='#6db858')

    # ax[0].set_ylabel('$C_P$ [-]')
    # ax[0].set_xlabel('Wind speed [m/s]')

    # ax[0].legend()

    # ax[1].plot(ref_ws,ref_ct,linestyle='solid',linewidth=2, label='Reference', color='grey')
    # ax[1].plot(wrf_ws,wrf_ct,linestyle='solid',linewidth=2, label='WRF', color='#fc1703')
    # ax[1].plot(ws,ConstantInduction['ct'],linestyle='--',linewidth=2, label='MITRotor: Constant Induction', color='#1a6e03')
    # ax[1].plot(ws,UnifiedMomentum['ct'],linestyle='--',linewidth=2, label='MITRotor: Unified Momentum', color='#3b8a25')
    # ax[1].plot(ws,MadsenMomentum['ct'],linestyle='--',linewidth=2, label='MITRotor: Madsen Momentum', color='#6db858')

    # ax[1].set_ylabel('$C_T$ [-]')
    # ax[1].set_xlabel('Wind speed [m/s]')

    # ax[1].legend()

    # plt.savefig("10MW_wrf_mitrotor_ref_NoLoss_TanInd.png", bbox_inches="tight", dpi=600) 