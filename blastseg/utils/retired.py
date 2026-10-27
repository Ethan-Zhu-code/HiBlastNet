"""Options from earlier research code that this release no longer implements.

Released checkpoints and their stored configurations still record these
options with their disabled values. They are accepted and ignored when they
are disabled, and rejected otherwise, so an unsupported setting can never be
silently dropped.
"""

ANY = object()  # parameter of a disabled feature; any recorded value is accepted

RETIRED_MODEL_OPTIONS = {
    "gcnet_detail_extra_blocks": 0,
    "gcnet_detail_highres_branch": False,
    "gcnet_highres_logit_refinement": False,
    "gcnet_highres_logit_refinement_channels": ANY,
    "gcnet_highres_logit_context": False,
    "gcnet_highres_refinement_gate": "none",
    "gcnet_highres_refinement_gate_channels": ANY,
    "gcnet_multiscale_leaf_support": False,
    "gcnet_lowres_context": False,
    "gcnet_lowres_context_channels": ANY,
    "gcnet_lowres_context_heads": ANY,
    "gcnet_hierarchical_semantic_fusion": False,
    "gcnet_hierarchical_fusion_channels": ANY,
    "gcnet_hlpc_enabled": False,
    "gcnet_hlpc_initial_scale": ANY,
    "gcnet_hlpc_temperature": ANY,
    "gcnet_hlpc_prior_logit_bias": ANY,
    "gcnet_hlpc_class_specific": ANY,
    "gcnet_hlpc_spatial_gate": ANY,
    "gcnet_hlpc_uncertainty_power": ANY,
    "gcnet_pcrr_enabled": False,
    "gcnet_pcrr_channels": ANY,
    "gcnet_pcrr_max_delta": ANY,
    "gcnet_pcrr_uncertainty_power": ANY,
    "gcnet_hcpd_enabled": False,
    "gcnet_hcpd_hidden_channels": ANY,
    "gcnet_hcpd_embedding_dim": ANY,
    "gcnet_cpmr_mode": "none",
    "gcnet_cpmr_channels": ANY,
    "gcnet_cpmr_source": ANY,
    "gcnet_csar_mode": "none",
    "gcnet_csar_channels": ANY,
    "gcnet_csar_max_delta": ANY,
    "gcnet_csar_propagation_iterations": ANY,
    "lesion_aux_source": "c6",
    "gcnet_center_aux_head": False,
    "gcnet_center_aux_detach": ANY,
    "leafwise_enabled": False,
    "leafwise_relation_dir": ANY,
    "leafwise_use_global_context": ANY,
    "leafwise_use_position_encoding": ANY,
    "leafwise_use_overlap_consistency": ANY,
    "leafwise_global_context_dim": ANY,
    "leafwise_aux_weight": ANY,
    "leafwise_overlap_weight": ANY,
    # All released models were trained from scratch.
    "pretrained_backbone_path": None,
    "pidnet_pretrained_path": None,
}


def pop_retired_options(options, retired=RETIRED_MODEL_OPTIONS):
    """Remove retired keys from ``options`` in place; raise if one is enabled."""
    for key in [key for key in options if key in retired]:
        value = options.pop(key)
        disabled = retired[key]
        if disabled is not ANY and value != disabled:
            raise ValueError(
                f"Option {key}={value!r} is not supported in this release "
                f"(only {disabled!r} is accepted)."
            )
    return options
