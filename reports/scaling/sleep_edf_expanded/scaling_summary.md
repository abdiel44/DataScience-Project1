# Scaling summary (Topic 5)

- Method: `standardize`
- Scaled columns (20): `['eeg_mean', 'eeg_std', 'eeg_var', 'eeg_rms', 'eeg_zero_crossing_rate', 'eeg_bandpower_delta', 'eeg_rel_power_delta', 'eeg_bandpower_theta', 'eeg_rel_power_theta', 'eeg_bandpower_alpha', 'eeg_rel_power_alpha', 'eeg_bandpower_beta', 'eeg_rel_power_beta', 'eeg_theta_alpha_ratio', 'eeg_spectral_entropy', 'sleep_stage_n1', 'sleep_stage_n2', 'sleep_stage_n3', 'sleep_stage_rem', 'sleep_stage_w']`
- Skipped columns (6): `['epoch_index', 'epoch_start_sample', 'epoch_end_sample', 'epoch_start_sec', 'epoch_end_sec', 'sfreq_hz']`

## Options

- exclude_columns: `['recording_id', 'subject_id', 'epoch_index', 'epoch_start_sample', 'epoch_end_sample', 'epoch_start_sec', 'epoch_end_sec', 'sfreq_hz', 'source_file', 'hypnogram_file', 'eeg_channel']`
- target_column (excluded): `'sleep_stage'`
- include_columns: `None`

