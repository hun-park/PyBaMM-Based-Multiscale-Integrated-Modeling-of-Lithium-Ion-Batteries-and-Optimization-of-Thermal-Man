import os, sys, pybamm
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

# SPM으로 시작
spm = pybamm.lithium_ion.SPM(options={'thermal': 'lumped', "lithium plating": "partially reversible",})
spm.default_parameter_values.search('Nominal cell capacity [A.h]')
spm.default_parameter_values.search('Upper voltage cut-off [V]')
spm.default_parameter_values.search('Lower voltage cut-off [V]')
print("------------------------------------")
param = pybamm.ParameterValues("OKane2022")
param.search('Nominal cell capacity [A.h]')
param.search('Upper voltage cut-off [V]')
param.search('Lower voltage cut-off [V]')

v_up = param['Upper voltage cut-off [V]']; v_low = param['Lower voltage cut-off [V]']; C = -param['Nominal cell capacity [A.h]']

protocol = 'CCCV'
# 1C CC-CV
globals()[f'{protocol}'] = pybamm.Experiment(
    [(
        pybamm.step.c_rate(-1, termination=f'{v_up}V'),
        pybamm.step.voltage(v_up, termination="C/100"),
        "Rest for 15 minutes",
        pybamm.step.c_rate(1, termination=f'{v_low}V'),
        "Rest for 15 minutes",
    )]
)

model, protocol = 'spm', 'CCCV'
globals()[f'sim_{model}_{protocol}'] = pybamm.Simulation(globals()[f'{model}'], experiment=globals()[f'{protocol}'], parameter_values=param, solver=solver)
globals()[f'sol_{model}_{protocol}'] = globals()[f'sim_{model}_{protocol}'].solve(initial_soc=0)
t = globals()[f'sol_{model}_{protocol}']['Time [s]'].entries
i = -globals()[f'sol_{model}_{protocol}']['Current [A]'].entries
v = globals()[f'sol_{model}_{protocol}']['Voltage [V]'].entries
T = globals()[f'sol_{model}_{protocol}']['Cell temperature [C]'].entries[0]
desired_t = [step.t_event[0] for step in globals()[f'sol_{model}_{protocol}'].cycles[0].steps if step.t_event is not None]
i_at_desired_t = -globals()[f'sol_{model}_{protocol}']['Current [A]'](desired_t)
v_at_desired_t = globals()[f'sol_{model}_{protocol}']['Voltage [V]'](desired_t)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 2))
ax1.plot(t, i); ax1.hlines(y=[-C], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax1.set_title('Current [A]')
ax1.plot(desired_t, i_at_desired_t, 'ro', markersize=8)
for dt, current_val in zip(desired_t, i_at_desired_t):
    ax1.text(dt, current_val, f"({dt:.1f}, {current_val:.2f})", color='red', fontsize=7, verticalalignment='bottom')
ax2.plot(t, v); ax2.hlines(y=[v_up, v_low], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax2.set_title('Voltage [V]')
ax2.plot(desired_t, v_at_desired_t, 'ro', markersize=8)
for dt, voltage_val in zip(desired_t, v_at_desired_t):
    ax2.text(dt, voltage_val, f"({dt:.1f}, {voltage_val:.2f})", color='red', fontsize=7, verticalalignment='bottom')
ax3.plot(t, T); ax3.hlines(y=[40], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax3.set_title('Temperature [C]')
plt.savefig(f'{model}_{protocol}.png', dpi=200)

protocol = 'MCCCV'
# Arbitary 3 stages MCC-CV
globals()[f'{protocol}'] = pybamm.Experiment(
    [(
        pybamm.step.c_rate(-3, termination=f'{v_up-0.4}V'),
        pybamm.step.c_rate(-2, termination=f'{v_up-0.2}V'),
        pybamm.step.c_rate(-1, termination=f'{v_up}V'),
        pybamm.step.voltage(v_up, termination="C/100"),
        "Rest for 15 minutes",
        pybamm.step.c_rate(1, termination=f'{v_low}V'),
        "Rest for 15 minutes",
    )]
)

model, protocol = 'spm', 'MCCCV'
globals()[f'sim_{model}_{protocol}'] = pybamm.Simulation(globals()[f'{model}'], experiment=globals()[f'{protocol}'], parameter_values=param, solver=solver)
globals()[f'sol_{model}_{protocol}'] = globals()[f'sim_{model}_{protocol}'].solve(initial_soc=0)
t = globals()[f'sol_{model}_{protocol}']['Time [s]'].entries
i = -globals()[f'sol_{model}_{protocol}']['Current [A]'].entries
v = globals()[f'sol_{model}_{protocol}']['Voltage [V]'].entries
T = globals()[f'sol_{model}_{protocol}']['Cell temperature [C]'].entries[0]
desired_t = [step.t_event[0] for step in globals()[f'sol_{model}_{protocol}'].cycles[0].steps if step.t_event is not None]
i_at_desired_t = -globals()[f'sol_{model}_{protocol}']['Current [A]'](desired_t)
v_at_desired_t = globals()[f'sol_{model}_{protocol}']['Voltage [V]'](desired_t)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 2))
ax1.plot(t, i); ax1.hlines(y=[-3*C, -2*C, -C], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax1.set_title('Current [A]')
ax1.plot(desired_t, i_at_desired_t, 'ro', markersize=8)
for dt, current_val in zip(desired_t, i_at_desired_t):
    ax1.text(dt, current_val, f"({dt:.1f}, {current_val:.2f})", color='red', fontsize=7, verticalalignment='bottom')
ax2.plot(t, v); ax2.hlines(y=[v_up-0.4, v_up-0.2, v_up, v_low], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax2.set_title('Voltage [V]')
ax2.plot(desired_t, v_at_desired_t, 'ro', markersize=8)
for dt, voltage_val in zip(desired_t, v_at_desired_t):
    ax2.text(dt, voltage_val, f"({dt:.1f}, {voltage_val:.2f})", color='red', fontsize=7, verticalalignment='bottom')
ax3.plot(t, T); ax3.hlines(y=[40], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax3.set_title('Temperature [C]')
plt.savefig(f'{model}_{protocol}.png', dpi=200)

model, protocol = 'spm', 'PC'
# pulse charging
pulse_duration = 10
globals()[f'{protocol}'] = []

globals()[f'{protocol}'].append(pybamm.Experiment(
    [(pybamm.step.c_rate(-1, duration=pulse_duration/2, termination=f'{v_up}V'), 
      pybamm.step.current(0, duration=pulse_duration/2, termination=f'{v_up}V'))]))

globals()[f'{protocol}'].append(pybamm.Experiment(
    [(pybamm.step.c_rate(-1, termination=f'{v_up}V'),
      pybamm.step.voltage(v_up, termination="C/100"),
      "Rest for 15 minutes",
      pybamm.step.c_rate(1, termination=f'{v_low}V'),
      "Rest for 15 minutes")]))

model, protocol = 'spm', 'PC'
globals()[f'sim_{model}_{protocol}'] = pybamm.Simulation(globals()[f'{model}'], experiment=globals()[f'{protocol}'][0], parameter_values=param, solver=solver)
globals()[f'sol_{model}_{protocol}'] = globals()[f'sim_{model}_{protocol}'].solve(initial_soc=0)

flag = 0
import tqdm
for i in tqdm.tqdm(range(600)):
    globals()[f'sim_{model}_{protocol}'] = pybamm.Simulation(globals()[f'{model}'], experiment=globals()[f'{protocol}'][0], parameter_values=param, solver=solver)
    globals()[f'sol_{model}_{protocol}'] = globals()[f'sim_{model}_{protocol}'].solve(starting_solution=globals()[f'sol_{model}_{protocol}'])
    flag = globals()[f'sol_{model}_{protocol}']['Voltage [V]'].entries[-1]
    
    if flag >= v_up:
        print(f'escape the loops with counts: {i}')
        break

globals()[f'sim_{model}_{protocol}'] = pybamm.Simulation(globals()[f'{model}'], experiment=globals()[f'{protocol}'][1], parameter_values=param, solver=solver)
globals()[f'sol_{model}_{protocol}'] = globals()[f'sim_{model}_{protocol}'].solve(starting_solution=globals()[f'sol_{model}_{protocol}'])

model, protocol = 'spm', 'PC'
t = globals()[f'sol_{model}_{protocol}']['Time [s]'].entries
i = -globals()[f'sol_{model}_{protocol}']['Current [A]'].entries
v = globals()[f'sol_{model}_{protocol}']['Voltage [V]'].entries
T = globals()[f'sol_{model}_{protocol}']['Cell temperature [C]'].entries[0]
desired_t = [step.t_event[0] for step in globals()[f'sol_{model}_{protocol}'].cycles[601].steps if step.t_event is not None]
i_at_desired_t = -globals()[f'sol_{model}_{protocol}']['Current [A]'](desired_t)
v_at_desired_t = globals()[f'sol_{model}_{protocol}']['Voltage [V]'](desired_t)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 2))
ax1.plot(t, i); ax1.hlines(y=[C], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax1.set_title('Current [A]')
ax1.plot(desired_t, i_at_desired_t, 'ro', markersize=8)
for dt, current_val in zip(desired_t, i_at_desired_t):
    ax1.text(dt, current_val, f"({dt:.1f}, {current_val:.2f})", color='red', fontsize=7, verticalalignment='bottom')
ax2.plot(t, v); ax2.hlines(y=[v_up, v_low], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax2.set_title('Voltage [V]')
ax2.plot(desired_t, v_at_desired_t, 'ro', markersize=8)
for dt, voltage_val in zip(desired_t, v_at_desired_t):
   ax2.text(dt, voltage_val, f"({dt:.1f}, {voltage_val:.2f})", color='red', fontsize=7, verticalalignment='bottom')
ax3.plot(t, T); ax3.hlines(y=[40], xmin=t[0], xmax=t[-1], colors='purple', linestyles='--', lw=1); ax3.set_title('Temperature [C]')
plt.savefig(f'{model}_{protocol}.png', dpi=200)
