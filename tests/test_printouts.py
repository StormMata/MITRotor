import numpy as np

from MITRotor import BEM, IEA15MW, BEMGeometry, NoTipLoss, HeckMomentum, ConstantInduction, ClassicalMomentum, NoTangentialInduction, AerodynamicProperties

if __name__ == "__main__":
    # Initialize rotor using the IEA10MW reference wind turbine model.
    rotor = IEA15MW()
    bem = BEM(rotor=rotor, geometry=BEMGeometry(Nr=53, Ntheta=120), tiploss_model=NoTipLoss(), momentum_model=ConstantInduction(), tangential_induction_model=NoTangentialInduction())

    # solve BEM for a control set point.
    pitch, tsr, yaw = np.deg2rad(0.085836909871247), 9.0552, np.deg2rad(0.0)
    sol = bem(pitch, tsr, yaw)

    # Print various quantities in BEM solution
    if sol.converged:
        print(f"BEM solution converged in {sol.niter} iterations.")
    else:
        print("BEM solution did NOT converge!")

    print(f"Control setpoints: {sol.pitch=:2.2f}, {sol.tsr=:2.2f}, {sol.yaw=:2.2f}")
    print(f"Power coefficient: {sol.Cp():2.3f}")
    print(f"Thrust coefficient: {sol.Ct():2.3f}")
    print(f"Local thrust coefficient: {sol.Ctprime():2.2f}")
    print(f"Axial induction: {sol.a():2.3f}")
    # print(f"Tangential induction: {sol.an():2.2f}")
    print(f"Rotor-effective windspeed: {sol.U():2.2f}")
    print(f"Far-wake streamwise velocity: {sol.u4():2.2f}")
    print(f"Far-wake lateral velocity: {sol.v4():2.2f}")
