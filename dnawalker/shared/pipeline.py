"""Architecture-independent prediction and refinement workflows.

Model modules retain their predictor construction, configuration adapters,
experimental-data loaders, plotting, and CLI surfaces.  They inject those
model-specific dependencies here so both architectures execute exactly the
same numerical control flow without hiding monkeypatchable module symbols.
"""


def run_prediction_refinement(
        predictor,
        config,
        model_input,
        experimental_curves,
        *,
        ensemble,
        noise_std,
        method,
        maxiter,
        multistart,
        seed,
        output_path,
        parameter_order_fn,
        ensemble_predict_fn,
        vector_to_param_dict_fn,
        build_name_map_fn,
        curve_rmse_fn,
        refine_fn,
        jitter_params_fn,
        write_params_fn,
        numpy_module):
    """Run the shared prediction, local-refinement, and reporting pipeline."""
    params = ensemble_predict_fn(
        predictor,
        model_input,
        ensemble=ensemble,
        noise_std=noise_std,
        seed=seed,
    )
    predicted = vector_to_param_dict_fn(
        params,
        predictor.get_param_names(),
    )

    name_map = build_name_map_fn(config.get_trainable_param_names())
    ranges = {
        name_map[name]: bounds
        for name, bounds in config.get_param_ranges().items()
    }
    log_params = config.get_log_transform_params()

    dl_rmse = curve_rmse_fn(predicted, experimental_curves)
    print(f"\n[1] DL 预测 RMSE: {dl_rmse:.4f}")

    best_params, best_rmse, _ = refine_fn(
        predicted,
        experimental_curves,
        ranges,
        method=method,
        maxiter=maxiter,
        verbose=True,
        log_params=log_params,
    )

    if multistart > 0:
        rng = numpy_module.random.default_rng(seed + 1)
        for restart_index in range(multistart):
            jittered = jitter_params_fn(
                predicted,
                ranges,
                rng,
                log_params=log_params,
            )
            refined, refined_rmse, _ = refine_fn(
                jittered,
                experimental_curves,
                ranges,
                method=method,
                maxiter=maxiter,
                verbose=False,
                log_params=log_params,
            )
            if refined_rmse < best_rmse:
                best_rmse, best_params = refined_rmse, refined
                print(
                    f"  [multistart {restart_index + 1}] "
                    f"新最优 RMSE: {refined_rmse:.4f}"
                )

    if (numpy_module.isfinite(dl_rmse)
            and dl_rmse > 0
            and numpy_module.isfinite(best_rmse)):
        improvement = f"{(dl_rmse - best_rmse) / dl_rmse * 100:.1f}%"
    elif dl_rmse == 0 and best_rmse == 0:
        improvement = "0.0%"
    else:
        improvement = "n/a"
    print(
        f"\n[2] 精修后最终 RMSE: {best_rmse:.4f}  "
        f"(DL→最终 降低 {improvement})"
    )
    print("\n最终参数:")
    for name in parameter_order_fn():
        print(f"  {name:<16} = {best_params[name]:.6e}")

    write_params_fn(best_params, output_path)
    print(f"\n已写入: {output_path}")
    return best_params, best_rmse, dl_rmse


def evaluate_refined_dataset(
        predictor,
        config,
        ranges,
        *,
        ensemble,
        noise_std,
        maxiter,
        multistart,
        seed,
        refine_on,
        load_model_input_fn,
        load_interpolated_curves_fn,
        load_raw_curves_fn,
        require_int_fn,
        ensemble_predict_fn,
        vector_to_param_dict_fn,
        rmse_fn,
        refine_fn,
        jitter_params_fn,
        numpy_module):
    """Evaluate one experimental dataset with optional physics refinement."""
    multistart = require_int_fn(multistart, "multistart", minimum=0)
    model_input = load_model_input_fn()
    _, experimental_curves = load_interpolated_curves_fn()

    params = ensemble_predict_fn(
        predictor,
        model_input,
        ensemble=ensemble,
        noise_std=noise_std,
        seed=seed,
    )
    predicted = vector_to_param_dict_fn(
        params,
        predictor.get_param_names(),
    )

    dl_rmse, _ = rmse_fn(predicted, experimental_curves)
    dl_avg = (
        float(dl_rmse.mean())
        if numpy_module.all(numpy_module.isfinite(dl_rmse))
        else float("inf")
    )

    best = predicted
    best_rmse = dl_avg
    if refine_on:
        log_params = config.get_log_transform_params()
        refined, refined_rmse, _ = refine_fn(
            predicted,
            experimental_curves,
            ranges,
            method="Powell",
            maxiter=maxiter,
            verbose=False,
            log_params=log_params,
        )
        if refined_rmse < best_rmse:
            best, best_rmse = refined, refined_rmse

        rng = numpy_module.random.default_rng(seed + 1)
        for _ in range(multistart):
            jittered = jitter_params_fn(
                predicted,
                ranges,
                rng,
                log_params=log_params,
            )
            refined, refined_rmse, _ = refine_fn(
                jittered,
                experimental_curves,
                ranges,
                method="Powell",
                maxiter=maxiter,
                verbose=False,
                log_params=log_params,
            )
            if refined_rmse < best_rmse:
                best, best_rmse = refined, refined_rmse

    final_rmse, simulated_signals = rmse_fn(best, experimental_curves)
    if simulated_signals is None:
        fam_rmse = tye_rmse = cy5_rmse = avg_rmse = float("inf")
    else:
        fam_rmse, tye_rmse, cy5_rmse = (
            float(final_rmse[0]),
            float(final_rmse[1]),
            float(final_rmse[2]),
        )
        avg_rmse = float(final_rmse.mean())

    time_raw, fam, tye, cy5 = load_raw_curves_fn()
    result = {
        "dl_avg": dl_avg,
        "fam_rmse": fam_rmse,
        "tye_rmse": tye_rmse,
        "cy5_rmse": cy5_rmse,
        "avg_rmse": avg_rmse,
        "params": best,
    }
    return result, simulated_signals, (time_raw, fam, tye, cy5)
