import numpy as np
import matplotlib.pyplot as plt
import pysteps

preds  = np.load("/home/yppatel/fusioncp/PreDiff/experiments/calibration/preds.npy")
full = np.load("/home/yppatel/fusioncp/PreDiff/experiments/calibration/full.npy")
pre_forecast = full[:,:7]

n_ensemble_members = 4
km_per_pixel = 0.5
pixels_per_km = 1 / km_per_pixel
timestep = 710

full_predictions = []
for precip in pre_forecast:
    velocity = pysteps.motion.darts.DARTS(precip)

    predictions = []
    for member in range(n_ensemble_members):
        perturbator = pysteps.noise.motion.initialize_bps(velocity, pixels_per_km, timestep)

        # HACK: not sure why, but the velocities perturbations are exceedingly large if not divided by 10 (units mismatch?)
        # this division seems to produce reasonable ensemble predictions, but we may wish to revisit this
        noise = pysteps.noise.motion.generate_bps(perturbator, .1) / 10

        ensemble_prediction = pysteps.nowcasts.extrapolation.forecast(precip[-1], velocity + noise, timesteps=6)
        predictions.append(ensemble_prediction)
    predictions = np.nan_to_num(np.array(predictions))

    full_predictions.append(predictions)
full_predictions = np.array(full_predictions)

print(full_predictions.shape)

np.save("steps.npy", full_predictions)