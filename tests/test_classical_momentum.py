import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from MITRotor import BEM, IEA15MW, BEMGeometry, NoTipLoss, ConstantInduction, NoTangentialInduction, BEMSolution

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


def plot_radial_distributions(sol: BEMSolution, save_to: Path):
    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].set_ylabel('$C_P$ [-]')
    axes[1].set_ylabel('$C_T$ [-]')
    axes[2].set_ylabel('$a_n$ [-]')

    # Classical momentum model

    axes[0].plot(sol.geom.mu, sol.Cp(grid="annulus"), label="[JYL]: avg: 0.537")
    axes[1].plot(sol.geom.mu, sol.Ct(grid="annulus"), label="[JYL]: avg: 0.818")
    axes[2].plot(sol.geom.mu, sol.a(grid="annulus"), label="[JYL]: avg: 0.287")

    r_sam  = np.loadtxt('/Users/stormmata/Downloads/debugging/r.csv', delimiter=',')
    cp_sam = np.loadtxt('/Users/stormmata/Downloads/debugging/cp.csv', delimiter=',')
    ct_sam = np.loadtxt('/Users/stormmata/Downloads/debugging/ct.csv', delimiter=',')
    a_sam  = np.loadtxt('/Users/stormmata/Downloads/debugging/a.csv', delimiter=',')

    axes[0].plot(r_sam, cp_sam, label="[SAM]: avg: 0.533; -0.74%")
    axes[1].plot(r_sam, ct_sam, label="[SAM]: avg: 0.814; -0.49%")
    axes[2].plot(r_sam, a_sam, label="[SAM]: avg: 0.284; -1.05%")

    axes[0].set_title('Classical Momentum Model')

    axes[2].set_ylim([0,0.5])

    [ax.legend(loc="lower center") for ax in axes]

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
    bem = BEM(rotor=rotor, geometry=BEMGeometry(Nr=53, Ntheta=120), tiploss_model=NoTipLoss(), momentum_model=ConstantInduction(), tangential_induction_model=NoTangentialInduction())

    # solve BEM for a control set point.
    pitch, tsr, yaw = np.deg2rad(0.085836909871247), 9.0552, np.deg2rad(0.0)
    sol = bem(pitch, tsr, yaw)

    # Plot
    plot_radial_distributions(sol, figdir / "ClassicalMomentum.png")