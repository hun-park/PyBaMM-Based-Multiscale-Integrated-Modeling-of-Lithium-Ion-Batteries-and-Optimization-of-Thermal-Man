import os, sys, pybamm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

solver = pybamm.CasadiSolver(
    mode="safe",
    rtol=1e-5,  # 상대 허용 오차 증가 → Solver가 더 유연해짐
    atol=1e-5,  # 절대 허용 오차 증가 → Solver가 작은 수치 변화에 덜 민감해짐
    root_method="hybr",  # 초기 조건 찾기 알고리즘을 SciPy의 "hybr"로 변경
    root_tol=1e-5,  # 루트 허용 오차 증가
    max_step_decrease_count=10,  # 스텝 크기 감소 허용 횟수를 증가시켜 Solver가 더 많은 시도 가능
    dt_max=100,  # 최대 시간 스텝을 100초로 줄여 수치적 안정성 증가
    extrap_tol=1e-6,  # 외삽 허용 오차 추가
    extra_options_setup={"max_num_steps": 100000, "print_stats": False},  # CasADi 설정 변경
    return_solution_if_failed_early=True,  # Solver가 실패해도 일부 결과 반환
    perturb_algebraic_initial_conditions=False,  # 초기 조건의 작은 교란 비활성화
    integrators_maxcount=200,  # 통합기 개수 증가 (메모리 사용 증가 가능)
)

# LAM은 이상함
# 둘 다 고려
# SPM으로 동작 시, "repeated recoverable right-hand side function errors."
# The electrolyte will deplete. 
# For the DFN, this means it will stop when the electrolyte gets close to zero (hence it will stop before reaching the 2.5V cut-off). 
# For the SPM, the simulations should run (once the dt_max is reduced) 
# but the results will make no physical sense as they do not include any electrolyte effect.
Okane_ageing_cycle = pybamm.Experiment(
    [ ("Charge at 1 C until 4.2 V", "Hold at 4.2 V until C/100", "Rest for 15 minutes", 
       "Discharge at 1 C until 2.5 V", "Rest for 15 minutes",) ] * 1000,
    termination="80% capacity",
    )

spm_cycle_aging = pybamm.lithium_ion.SPMe(options={'thermal': 'lumped', 
                                               "lithium plating": "partially reversible",
                                               "SEI": "solvent-diffusion limited"})
param = pybamm.ParameterValues("OKane2022")
sim_cycle_aging = pybamm.Simulation(spm_cycle_aging, parameter_values=param, experiment=Okane_ageing_cycle, solver=solver)
sol_cycle_aging = sim_cycle_aging.solve();
param.search('Nominal cell capacity [A.h]')
param.search('Upper voltage cut-off [V]')
param.search('Lower voltage cut-off [V]')
print("Okane_ageing_cycle, SPMe")
print(sol_cycle_aging.solve_time)
print(len(sol_cycle_aging.cycles))

# 결과확인 : 전부 발생
cycle_num_1 = 10
t_wT = sol_cycle_aging.cycles[cycle_num_1]['Time [s]'].entries; i_wT = sol_cycle_aging.cycles[cycle_num_1]['Current [A]'].entries; v_wT = sol_cycle_aging.cycles[cycle_num_1]['Voltage [V]'].entries; T = sol_cycle_aging.cycles[cycle_num_1]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_aging.cycles[cycle_num_1]['Time [s]'].entries))
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}: Current [A]')
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}: Voltage [V]')
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}: Cell temperature [C]')

cycle_num_2 = 10
t_wT = sol_cycle_aging.cycles[cycle_num_2]['Time [s]'].entries; i_wT = sol_cycle_aging.cycles[cycle_num_2]['Current [A]'].entries; v_wT = sol_cycle_aging.cycles[cycle_num_2]['Voltage [V]'].entries; T = sol_cycle_aging.cycles[cycle_num_2]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_aging.cycles[cycle_num_2]['Time [s]'].entries))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Current [A]'); ax1.legend(["10", "900"], loc="upper left")
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Voltage [V]'); ax2.legend(["10", "900"], loc="upper left")
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Cell temperature [C]'); ax3.legend(["10", "900"], loc="upper left")
plt.savefig(f'Okane_ageing_cycle_SPMe_{cycle_num_1}_{cycle_num_2}.png', dpi=200)

t_wT = sol_cycle_aging.summary_variables['Cycle number'];
c_wT = sol_cycle_aging.summary_variables['Capacity [A.h]'];
SEI_wT = sol_cycle_aging.summary_variables['Change in loss of capacity to negative SEI [A.h]'];
li_wT = sol_cycle_aging.summary_variables['Change in loss of capacity to negative lithium plating [A.h]']
t_wT = range(len(sol_cycle_aging.summary_variables['Cycle number']))
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, c_wT); ax1.set_ylim([0, 6]); ax1.set_title(f'Capacity [A.h]')
ax2.plot(t_wT, SEI_wT); ax2.set_title(f'Capacity loss (SEI) [A.h]')
ax3.plot(t_wT, li_wT); ax3.set_title(f'Capacity loss (lithium plating) [A.h]')
plt.savefig(f'Okane_ageing_cycle_SPMe_summary_variables.png', dpi=200)

# High C-rates above 1C
Okane_high_ageing_cycle = pybamm.Experiment(
    [ ("Charge at 3 C until 4.1 V", "Hold at 4.1 V until C/100", "Rest for 15 minutes", 
       "Discharge at 1 C until 2.7 V", "Rest for 15 minutes",) ] * 1000,
    termination="80% capacity",
    )

sim_cycle_3C = pybamm.Simulation(spm_cycle_aging, parameter_values=param, experiment=Okane_high_ageing_cycle, solver=solver)
sol_cycle_3C = sim_cycle_3C.solve();
print("Okane_high_ageing_cycle, SPMe")
print(sol_cycle_3C.solve_time)

# 결과확인
cycle_num_1 = 10
t_wT = sol_cycle_3C.cycles[cycle_num_1]['Time [s]'].entries; i_wT = sol_cycle_3C.cycles[cycle_num_1]['Current [A]'].entries; v_wT = sol_cycle_3C.cycles[cycle_num_1]['Voltage [V]'].entries; T = sol_cycle_3C.cycles[cycle_num_1]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_3C.cycles[cycle_num_1]['Time [s]'].entries))
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}: Current [A]')
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}: Voltage [V]')
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}: Cell temperature [C]')

cycle_num_2 = 900
t_wT = sol_cycle_3C.cycles[cycle_num_2]['Time [s]'].entries; i_wT = sol_cycle_3C.cycles[cycle_num_2]['Current [A]'].entries; v_wT = sol_cycle_3C.cycles[cycle_num_2]['Voltage [V]'].entries; T = sol_cycle_3C.cycles[cycle_num_2]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_3C.cycles[cycle_num_2]['Time [s]'].entries))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Current [A]'); ax1.legend(["10", "900"], loc="upper left")
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Voltage [V]'); ax2.legend(["10", "900"], loc="upper left")
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Cell temperature [C]'); ax3.legend(["10", "900"], loc="upper left")
plt.savefig(f'Okane_high_ageing_cycle_SPMe_{cycle_num_1}_{cycle_num_2}.png', dpi=200)

t_wT = sol_cycle_3C.summary_variables['Cycle number'];
c_wT = sol_cycle_3C.summary_variables['Capacity [A.h]'];
SEI_wT = sol_cycle_3C.summary_variables['Change in loss of capacity to negative SEI [A.h]'];
li_wT = sol_cycle_3C.summary_variables['Change in loss of capacity to negative lithium plating [A.h]']
t_wT = range(len(sol_cycle_3C.summary_variables['Cycle number']))
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, c_wT); ax1.set_ylim([0, 6]); ax1.set_title(f'Capacity [A.h]')
ax2.plot(t_wT, SEI_wT); ax2.set_title(f'Capacity loss (SEI) [A.h]')
ax3.plot(t_wT, li_wT); ax3.set_title(f'Capacity loss (lithium plating) [A.h]')
plt.savefig(f'Okane_high_ageing_cycl_SPMe_summary_variables.png', dpi=200)

# SPMe
spme_cycle_3C = pybamm.lithium_ion.SPMe(options={'thermal': 'lumped', 
                                               "lithium plating": "partially reversible", "lithium plating porosity change": "true",
                                               "loss of active material": "stress and reaction-driven",
                                               "SEI": "solvent-diffusion limited", "SEI on cracks": "true", "SEI porosity change": "true"})
param = pybamm.ParameterValues("OKane2022")
sim_cycle_spme = pybamm.Simulation(spme_cycle_3C, parameter_values=param, experiment=Okane_high_ageing_cycle, solver=solver)
sol_cycle_spme = sim_cycle_spme.solve();
print("Okane_high_ageing_cycle, SPMe_3C")
print(sol_cycle_spme.solve_time)

# 결과확인
cycle_num_1 = 10
t_wT = sol_cycle_spme.cycles[cycle_num_1]['Time [s]'].entries; i_wT = sol_cycle_spme.cycles[cycle_num_1]['Current [A]'].entries; v_wT = sol_cycle_spme.cycles[cycle_num_1]['Voltage [V]'].entries; T = sol_cycle_spme.cycles[cycle_num_1]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_spme.cycles[cycle_num_1]['Time [s]'].entries))
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}: Current [A]')
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}: Voltage [V]')
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}: Cell temperature [C]')

cycle_num_2 = 900
t_wT = sol_cycle_spme.cycles[cycle_num_2]['Time [s]'].entries; i_wT = sol_cycle_spme.cycles[cycle_num_2]['Current [A]'].entries; v_wT = sol_cycle_spme.cycles[cycle_num_2]['Voltage [V]'].entries; T = sol_cycle_spme.cycles[cycle_num_2]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_spme.cycles[cycle_num_2]['Time [s]'].entries))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Current [A]'); ax1.legend(["10", "900"], loc="upper left")
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Voltage [V]'); ax2.legend(["10", "900"], loc="upper left")
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Cell temperature [C]'); ax3.legend(["10", "900"], loc="upper left")
plt.savefig(f'Okane_high_ageing_cycle_SPMe_3C_{cycle_num_1}_{cycle_num_2}.png', dpi=200)

t_wT = sol_cycle_spme.summary_variables['Cycle number'];
c_wT = sol_cycle_spme.summary_variables['Capacity [A.h]'];
SEI_wT = sol_cycle_spme.summary_variables['Change in loss of capacity to negative SEI [A.h]'];
li_wT = sol_cycle_spme.summary_variables['Change in loss of capacity to negative lithium plating [A.h]']
t_wT = range(len(sol_cycle_spme.summary_variables['Cycle number']))
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, c_wT); ax1.set_ylim([0, 6]); ax1.set_title(f'Capacity [A.h]')
ax2.plot(t_wT, SEI_wT); ax2.set_title(f'Capacity loss (SEI) [A.h]')
ax3.plot(t_wT, li_wT); ax3.set_title(f'Capacity loss (lithium plating) [A.h]')
plt.savefig(f'Okane_high_ageing_cycl_SPMe_3C_summary_variables.png', dpi=200)

# DFN
DFN_cycle_3C = pybamm.lithium_ion.DFN(options={'thermal': 'lumped', 
                                               "lithium plating": "partially reversible", "lithium plating porosity change": "true",
                                               "loss of active material": "stress and reaction-driven",
                                               "SEI": "solvent-diffusion limited", "SEI on cracks": "true", "SEI porosity change": "true"})
param = pybamm.ParameterValues("OKane2022")
sim_cycle_dfn = pybamm.Simulation(DFN_cycle_3C, parameter_values=param, experiment=Okane_high_ageing_cycle, solver=solver)
sol_cycle_dfn = sim_cycle_dfn.solve();
print("Okane_high_ageing_cycle, DFN")
print(sol_cycle_dfn.solve_time)

# 결과확인
cycle_num_1 = 10
t_wT = sol_cycle_dfn.cycles[cycle_num_1]['Time [s]'].entries; i_wT = sol_cycle_dfn.cycles[cycle_num_1]['Current [A]'].entries; v_wT = sol_cycle_dfn.cycles[cycle_num_1]['Voltage [V]'].entries; T = sol_cycle_dfn.cycles[cycle_num_1]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_dfn.cycles[cycle_num_1]['Time [s]'].entries))
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}: Current [A]')
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}: Voltage [V]')
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}: Cell temperature [C]')

cycle_num_2 = 900
t_wT = sol_cycle_dfn.cycles[cycle_num_2]['Time [s]'].entries; i_wT = sol_cycle_dfn.cycles[cycle_num_2]['Current [A]'].entries; v_wT = sol_cycle_dfn.cycles[cycle_num_2]['Voltage [V]'].entries; T = sol_cycle_dfn.cycles[cycle_num_2]['Cell temperature [C]'].entries[0]
t_wT = range(len(sol_cycle_dfn.cycles[cycle_num_2]['Time [s]'].entries))
ax1.plot(t_wT, i_wT); ax1.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Current [A]'); ax1.legend(["10", "900"], loc="upper left")
ax2.plot(t_wT, v_wT); ax2.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Voltage [V]'); ax2.legend(["10", "900"], loc="upper left")
ax3.plot(t_wT, T); ax3.set_title(f'Cycle Num. {cycle_num_1}&{cycle_num_2} : Cell temperature [C]'); ax3.legend(["10", "900"], loc="upper left")
plt.savefig(f'Okane_high_ageing_cycle_DFN_3C_{cycle_num_1}_{cycle_num_2}.png', dpi=200)

t_wT = sol_cycle_dfn.summary_variables['Cycle number'];
c_wT = sol_cycle_dfn.summary_variables['Capacity [A.h]'];
SEI_wT = sol_cycle_dfn.summary_variables['Change in loss of capacity to negative SEI [A.h]'];
li_wT = sol_cycle_dfn.summary_variables['Change in loss of capacity to negative lithium plating [A.h]']
t_wT = range(len(sol_cycle_dfn.summary_variables['Cycle number']))
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,2))
ax1.plot(t_wT, c_wT); ax1.set_ylim([0, 6]); ax1.set_title(f'Capacity [A.h]')
ax2.plot(t_wT, SEI_wT); ax2.set_title(f'Capacity loss (SEI) [A.h]')
ax3.plot(t_wT, li_wT); ax3.set_title(f'Capacity loss (lithium plating) [A.h]')
plt.savefig(f'Okane_high_ageing_cycl_DFN_3C_summary_variables.png', dpi=200)