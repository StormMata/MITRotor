import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from MITRotor import BEM, IEA10MW, IEA15MW, BEMGeometry, AerodynamicProperties, NoTipLoss, PrandtlTipLoss, ConstantInduction, ClassicalMomentum, UnifiedMomentum, MadsenMomentum, NoTangentialInduction, DefaultTangentialInduction, BEMSolution

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

    bem = BEM(rotor=rotor, geometry=BEMGeometry(Nr=50, Ntheta=160),tiploss_model=NoTipLoss(), momentum_model=ConstantInduction(), tangential_induction_model=NoTangentialInduction())

    # aerodynamic_model=AerodynamicProperties(U=1)

    # ws        = [4.75, 6, 7, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12]
    # pitch_wrf = [1.667, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.596, 4.508, 5.953]
    # tsr_wrf   = [13.170, 10.639, 10.634, 10.635, 10.057, 9.531, 9.054, 8.623, 8.231, 7.873, 7.545]

    pitch_wrf = [0.0]
    tsr_wrf   = [9.05]

    mit_cp = np.zeros_like(pitch_wrf)
    mit_ct = np.zeros_like(pitch_wrf)

    # for i in range(len(pitch_wrf)):
    for i in range(1):


        # solve BEM for a control set point.
        # pitch, tsr, yaw = np.deg2rad(pitch_wrf[i]), tsr_wrf[i], np.deg2rad(0.0)
        # sol = bem(pitch, tsr, yaw, U=1.0)

        pitch, tsr, yaw = np.deg2rad(0.522), 8.882, np.deg2rad(0.0)
        sol = bem(pitch, tsr, yaw, U=np.array([0,1,2,3,4,5]))

        # Plot
        # plot_radial_distributions(sol, figdir / "ConstantInduction_15MW_test.png")
        mit_cp[i] = sol.Cp()
        mit_ct[i] = sol.Ct()

    # np.save("MadsenMomentum_NoLoss_TanInd.npy", {"cp": mit_cp, "ct": mit_ct})
    # print("Attributes and methods in sol:")
    # print(dir(sol))

    # print("sol attributes and values:")
    # for key, value in sol.__dict__.items():
    #     print(f"{key}: {value}")

    print(sol.Cp())

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