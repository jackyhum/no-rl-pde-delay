from gymnasium.envs.registration import register

register(
    id="PDEControlGym-TransportPDE1D", entry_point="pde_control_gym.src:TransportPDE1D"
)

register(
    id="PDEControlGym-ReactionDiffusionPDE1D", entry_point="pde_control_gym.src:ReactionDiffusionPDE1D"
)

register(
    id="PDEControlGym-NavierStokes2D", entry_point="pde_control_gym.src:NavierStokes2D"
)


register(
    id="PDEControlGym-SpatialDelayPDE", entry_point="pde_control_gym.src:SpatialDelayPDE"
)
