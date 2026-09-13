"""CP-SAT replay of the original static decoder, plus joint slot neighborhoods.

Integer predictions never enter the Pareto archive. Every returned chromosome
is checked by independent integer replay and the unmodified float decoder.
"""
from __future__ import annotations
import math
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'lib/base'
sys.path.insert(0, str(BASE))
from ortools.sat.python import cp_model
from src.algorithm.nsga3.encoding import Chromosome

TIME_SCALE = 100_000
POWER_SCALE = 10_000
ENERGY_SCALE = TIME_SCALE * POWER_SCALE


def quantize(x, scale):
    return int(round(float(x)*scale))


def validate(chrom):
    inst = chrom.instance
    n = inst.total_operations
    if min(inst.num_jobs, inst.num_machines, inst.num_agv, inst.num_speeds, n) < 1:
        raise ValueError('Positive static dimensions required')
    if any(x <= 0 for x in inst.num_operations):
        raise ValueError('Every static job requires an operation')
    for name in ('os','ma','agv_assign','agv_speed'):
        a = np.asarray(getattr(chrom,name))
        if a.shape != (n,) or a.dtype.kind not in 'iu':
            raise ValueError('Invalid integer gene: '+name)
    if sorted(chrom.os.tolist()) != [i for i in range(inst.num_jobs)
                                       for _ in range(inst.num_operations[i])]:
        raise ValueError('OS multiplicities do not describe the static instance')
    keys = [(i,j) for i in range(inst.num_jobs) for j in range(inst.num_operations[i])]
    for f,(i,j) in enumerate(keys):
        machines = inst.get_compatible_machines(i,j)
        if not machines or len(set(machines)) != len(machines):
            raise ValueError('Empty/duplicate compatibility list')
        if any(not 0 <= u < inst.num_machines for u in machines):
            raise ValueError('Machine ID outside domain')
        if not 0 <= int(chrom.ma[f]) < len(machines):
            raise ValueError('MA index outside domain (no modulo repair)')
        for u in machines:
            p = inst.get_processing_time(i,j,u)
            if not np.isfinite(p) or p <= 0:
                raise ValueError('Finite positive processing times required')
    for name,upper in [('agv_assign',inst.num_agv),('agv_speed',inst.num_speeds)]:
        if np.any(getattr(chrom,name)<0) or np.any(getattr(chrom,name)>=upper):
            raise ValueError('Invalid '+name+' (no modulo repair)')
    for name in ('machine_proc_power','machine_idle_power','machine_setup_power','machine_setup_time'):
        a = np.asarray(getattr(inst,name),float)
        if a.shape != (inst.num_machines,) or not np.isfinite(a).all() or (a<0).any():
            raise ValueError('Explicit nonnegative machine parameters required: '+name)
    d = np.asarray(inst.distance_matrix,float)
    if d.shape != (inst.num_machines+1,)*2 or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Explicit finite nonnegative distances required')
    if any(not np.isfinite(v) or v <= 0 for v in inst.speed_levels):
        raise ValueError('Finite positive speeds required')
    if any(inst.agv_load_power(v)<0 or inst.agv_empty_power(v)<0 for v in inst.speed_levels):
        raise ValueError('Nonnegative transport power required')
    return keys


def flat_order(chrom):
    keys = validate(chrom)
    lookup = {op:f for f,op in enumerate(keys)}
    counts = [0]*chrom.instance.num_jobs
    order = []
    for job in chrom.os:
        job = int(job)
        order.append(lookup[job,counts[job]])
        counts[job] += 1
    return keys,order


def integer_replay(chrom):
    """Independent scalar implementation of rounded primitive semantics."""
    keys, order = flat_order(chrom)
    inst = chrom.instance
    m,a,j = inst.num_machines,inst.num_agv,inst.num_jobs
    ma,aa,ja = [0]*m,[0]*a,[0]*j
    aloc,jloc,lastjob = [0]*a,[0]*j,[-1]*m
    workload = [0]*m
    energy = 0
    trace = {}
    for f in order:
        job,op = keys[f]
        u = inst.get_compatible_machines(job,op)[int(chrom.ma[f])]
        av,sv = int(chrom.agv_assign[f]),int(chrom.agv_speed[f])
        speed = inst.speed_levels[sv]
        origin,dest = jloc[job],u+1
        de,dl = inst.get_distance(aloc[av],origin),inst.get_distance(origin,dest)
        empty = quantize(de/speed,TIME_SCALE)
        loaded = quantize(dl/speed,TIME_SCALE)
        change = lastjob[u] >= 0 and lastjob[u] != job
        setup = quantize(inst.machine_setup_time[u],TIME_SCALE) if change else 0
        proc = quantize(inst.get_processing_time(job,op,u),TIME_SCALE)
        pickup = max(ja[job],aa[av]+empty)
        arrival = pickup+loaded
        start = max(arrival,ma[u]+setup)
        end = start+proc
        idle = start-ma[u]-setup if lastjob[u] >= 0 else 0
        assert idle >= 0
        energy += quantize(inst.get_processing_time(job,op,u)*inst.machine_proc_power[u],ENERGY_SCALE)
        if change:
            energy += quantize(inst.machine_setup_time[u]*inst.machine_setup_power[u],ENERGY_SCALE)
        energy += idle*quantize(inst.machine_idle_power[u],POWER_SCALE)
        energy += quantize(inst.transport_energy(de,speed,False),ENERGY_SCALE)
        energy += quantize(inst.transport_energy(dl,speed,True),ENERGY_SCALE)
        ma[u],aa[av],ja[job] = end,arrival,end
        aloc[av],jloc[job],lastjob[u] = dest,dest,job
        workload[u] += proc
        trace[f] = [pickup,arrival,start,end]
    return dict(objectives=[max(ma),energy,max(workload)-min(workload)],trace=trace)


class Kernel:
    """Small variable cluster with all remaining timing propagated exactly on grid."""
    def __init__(self,parent,cluster=(),objective=0,epsilon=None,
                 families=('order','machine','agv','speed')):
        self.started = time.monotonic()
        self.parent = parent
        self.keys,self.order = flat_order(parent)
        self.inst = inst = parent.instance
        self.n = n = len(self.keys)
        cluster = [int(x) for x in cluster]
        if len(set(cluster)) != len(cluster) or any(not 0<=x<n for x in cluster):
            raise ValueError('Invalid cluster operation IDs')
        if objective not in (0,1,2):
            raise ValueError('Objective index must be 0, 1 or 2')
        if not set(families) <= {'order','machine','agv','speed'}:
            raise ValueError('Unknown decision family')
        self.cluster = frozenset(cluster)
        self.families = frozenset(families)
        self.objective = objective
        self.epsilon = epsilon
        self.model = cp_model.CpModel()
        self.bounds = {}
        m,a,j,s = inst.num_machines,inst.num_agv,inst.num_jobs,inst.num_speeds
        maxproc = max(inst.processing_times.values())
        maxtravel = float(np.max(inst.distance_matrix))/min(inst.speed_levels)
        self.horizon = H = int(math.ceil(n*(maxproc+2*maxtravel+max(inst.machine_setup_time)+1)*TIME_SCALE))
        setup_ticks = [quantize(x,TIME_SCALE) for x in inst.machine_setup_time]
        setup_energy = [quantize(x*y,ENERGY_SCALE) for x,y in zip(inst.machine_setup_time,inst.machine_setup_power)]
        idle_power = [quantize(x,POWER_SCALE) for x in inst.machine_idle_power]
        travel_time,load_energy,empty_energy = [],[],[]
        for u in range(m+1):
            for v in range(m+1):
                for speed in inst.speed_levels:
                    d = inst.get_distance(u,v)
                    travel_time.append(quantize(d/speed,TIME_SCALE))
                    load_energy.append(quantize(inst.transport_energy(d,speed,True),ENERGY_SCALE))
                    empty_energy.append(quantize(inst.transport_energy(d,speed,False),ENERGY_SCALE))
        proc_emax = quantize(maxproc*max(inst.machine_proc_power),ENERGY_SCALE)+1
        E = n*(proc_emax+max(setup_energy)+H*max(idle_power)+max(load_energy)+max(empty_energy)+1)
        if E > 10**17:
            raise ValueError('Conservative integer range exceeds engineering safety cap')
        ma,aa,ja = [0]*m,[0]*a,[0]*j
        aloc,jloc,mjob = [0]*a,[0]*j,[-1]*m
        workloads = [[] for _ in range(m)]
        energies = []
        self.rows = []
        self.trace = []
        opvars = []
        for p,f in enumerate(self.order):
            candidates = sorted(self.cluster) if f in self.cluster and 'order' in self.families else [f]
            rows = []
            for of in candidates:
                oi,oj = self.keys[of]
                machines = inst.get_compatible_machines(oi,oj)
                choices = range(len(machines)) if of in self.cluster and 'machine' in self.families else [int(parent.ma[of])]
                for mi in choices:
                    u = machines[mi]
                    pt = inst.get_processing_time(oi,oj,u)
                    rows.append([of,oi,mi,u,quantize(pt,TIME_SCALE),quantize(pt*inst.machine_proc_power[u],ENERGY_SCALE)])
            attrs = [self.domain([r[k] for r in rows],f'p{p}_attr{k}') for k in range(6)]
            self.model.add_allowed_assignments(attrs,rows)
            opv,job,mi,u,proc,ep = attrs
            av = self.domain(range(a),f'p{p}_agv') if f in self.cluster and 'agv' in self.families else self.element(opv,[int(x) for x in parent.agv_assign],f'p{p}_agv')
            sv = self.domain(range(s),f'p{p}_speed') if f in self.cluster and 'speed' in self.families else self.element(opv,[int(x) for x in parent.agv_speed],f'p{p}_speed')
            opvars.append(opv)
            self.rows.append((opv,mi,av,sv))
            for v,hint in [(opv,f),(mi,int(parent.ma[f])),(av,int(parent.agv_assign[f])),(sv,int(parent.agv_speed[f]))]:
                if not isinstance(v,int):
                    self.model.add_hint(v,hint)
            cargo = self.element(job,ja,f'p{p}_cargo')
            origin = self.element(job,jloc,f'p{p}_origin')
            oldloc = self.element(av,aloc,f'p{p}_agvloc')
            oldend = self.element(av,aa,f'p{p}_agvend')
            mend = self.element(u,ma,f'p{p}_mend')
            prevjob = self.element(u,mjob,f'p{p}_prevjob')
            dest = self.linear(u+1,1,m,f'p{p}_dest')
            def trip(fromloc,toloc,values,label):
                index = self.linear((fromloc*(m+1)+toloc)*s+sv,0,len(values)-1,f'p{p}_{label}idx')
                return self.element(index,values,f'p{p}_{label}')
            empty = trip(oldloc,origin,travel_time,'empty')
            loaded = trip(origin,dest,travel_time,'loaded')
            ee = trip(oldloc,origin,empty_energy,'ee')
            el = trip(origin,dest,load_energy,'el')
            used = self.boolean_ge(prevjob,0,f'p{p}_used')
            same = self.boolean_eq(prevjob,job,f'p{p}_samejob')
            change = self.model.new_bool_var(f'p{p}_change')
            self.model.add(change<=used)
            self.model.add(change+same<=1)
            self.model.add(change>=used-same)
            self.bounds[change.index]=(0,1)
            setup = self.choose(change,self.element(u,setup_ticks,f'p{p}_st'),0,f'p{p}_setup')
            es = self.choose(change,self.element(u,setup_energy,f'p{p}_se'),0,f'p{p}_es')
            pickup = self.maximum([cargo,oldend+empty],0,H,f'p{p}_pickup')
            arrival = self.linear(pickup+loaded,0,H,f'p{p}_arrival')
            start = self.maximum([arrival,mend+setup],0,H,f'p{p}_start')
            end = self.linear(start+proc,0,H,f'p{p}_end')
            gap = self.linear(start-mend-setup,0,H,f'p{p}_gap')
            idle = self.choose(used,gap,0,f'p{p}_idle')
            possible_m = sorted(set(r[3] for r in rows))
            ei = self.scaled_by_choice(idle,u,idle_power,possible_m,f'p{p}_idleenergy')
            energies.extend([ep,es,ee,el,ei])
            self.trace.append([pickup,arrival,start,end])
            # Outside cluster states remain symbolic when affected upstream.
            possible_m = sorted(set(r[3] for r in rows))
            possible_j = sorted(set(r[1] for r in rows))
            for machine in possible_m:
                hit = self.boolean_eq(u,machine,f'p{p}_hitm{machine}')
                ma[machine] = self.choose(hit,end,ma[machine],f'p{p}_ma{machine}')
                mjob[machine] = self.choose(hit,job,mjob[machine],f'p{p}_mj{machine}')
                workloads[machine].append(self.choose(hit,proc,0,f'p{p}_work{machine}'))
            possible_a = range(a) if not isinstance(av,int) else [av]
            for agv in possible_a:
                hit = self.boolean_eq(av,agv,f'p{p}_hita{agv}')
                aa[agv] = self.choose(hit,arrival,aa[agv],f'p{p}_aa{agv}')
                aloc[agv] = self.choose(hit,dest,aloc[agv],f'p{p}_al{agv}')
            for ji in possible_j:
                hit = self.boolean_eq(job,ji,f'p{p}_hitj{ji}')
                ja[ji] = self.choose(hit,end,ja[ji],f'p{p}_ja{ji}')
                jloc[ji] = self.choose(hit,dest,jloc[ji],f'p{p}_jl{ji}')
        positions = [self.var(0,n-1,f'position{f}') for f in range(n)]
        self.model.add_inverse(opvars,positions)
        key_to_f = {op:f for f,op in enumerate(self.keys)}
        for f,(ji,oi) in enumerate(self.keys):
            if oi:
                self.model.add(positions[key_to_f[ji,oi-1]]<positions[f])
        loads = [self.linear(sum(w),0,H,f'workload{u}') for u,w in enumerate(workloads)]
        hi = self.maximum(loads,0,H,'maxload')
        lo = self.var(0,H,'minload')
        self.model.add_min_equality(lo,loads)
        self.objectives = [self.maximum(ma,0,H,'cmax'),self.linear(sum(energies),0,E,'energy'),
                           self.linear(hi-lo,0,H,'balance')]
        if epsilon is not None:
            if len(epsilon)!=3 or any(v is not None and (not math.isfinite(v) or v<0) for v in epsilon):
                raise ValueError('Three nonnegative epsilon bounds required')
            for k,(v,sc) in enumerate(zip(epsilon,[TIME_SCALE,ENERGY_SCALE,TIME_SCALE])):
                if v is not None:
                    self.model.add(self.objectives[k]<=int(math.floor(v*sc)))
        self.model.minimize(self.objectives[objective])
        invalid = self.model.validate()
        if invalid:
            raise ValueError('CP model invalid: '+invalid)
        self.build_seconds = time.monotonic()-self.started

    def var(self,lo,hi,name):
        v = self.model.new_int_var(int(lo),int(hi),name)
        self.bounds[v.index]=(int(lo),int(hi))
        return v

    def scaled_by_choice(self,x,index,constants,choices,name):
        """Exact constant multiplication selected by machine; no int_prod."""
        hi = self.bound(x)[1]*max(constants[k] for k in choices)
        if isinstance(index,int):
            return self.linear(x*constants[index],0,hi,name)
        value = self.var(0,hi,name)
        for k in choices:
            hit = self.boolean_eq(index,k,name+f'_machine{k}')
            self.model.add(value==x*constants[k]).only_enforce_if(hit)
        return value

    def domain(self,values,name):
        values = sorted(set(int(x) for x in values))
        if len(values)==1:
            return values[0]
        v = self.model.new_int_var_from_domain(cp_model.Domain.from_values(values),name)
        self.bounds[v.index]=(values[0],values[-1])
        return v

    def bound(self,v):
        return (v,v) if isinstance(v,int) else self.bounds[v.index]

    def linear(self,expr,lo,hi,name):
        if isinstance(expr,int):
            if not lo<=expr<=hi:
                raise ValueError('Constant outside domain: '+name)
            return expr
        v = self.var(lo,hi,name)
        self.model.add(v==expr)
        return v

    def element(self,index,values,name):
        if isinstance(index,int):
            return values[index]
        v = self.var(min(self.bound(x)[0] for x in values),max(self.bound(x)[1] for x in values),name)
        self.model.add_element(index,values,v)
        return v

    def boolean_eq(self,left,right,name):
        if isinstance(left,int) and isinstance(right,int):
            return int(left==right)
        b = self.var(0,1,name)
        self.model.add(left==right).only_enforce_if(b)
        self.model.add(left!=right).only_enforce_if(b.Not())
        return b

    def boolean_ge(self,left,right,name):
        if isinstance(left,int):
            return int(left>=right)
        b = self.var(0,1,name)
        self.model.add(left>=right).only_enforce_if(b)
        self.model.add(left<right).only_enforce_if(b.Not())
        return b

    def choose(self,condition,yes,no,name):
        if isinstance(condition,int):
            return yes if condition else no
        lo = min(self.bound(yes)[0],self.bound(no)[0])
        hi = max(self.bound(yes)[1],self.bound(no)[1])
        v = self.var(lo,hi,name)
        self.model.add(v==yes).only_enforce_if(condition)
        self.model.add(v==no).only_enforce_if(condition.Not())
        return v

    def maximum(self,exprs,lo,hi,name):
        if all(isinstance(x,int) for x in exprs):
            return max(exprs)
        v = self.var(lo,hi,name)
        self.model.add_max_equality(v,exprs)
        return v

    def solve(self,seconds=5.,seed=9080801,recovery_path=None,recovery_context=None,deadline=None):
        if not 0 < seconds <= 120:
            raise ValueError('Engineering solve cap must be in (0,120] seconds')
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = float(seconds)
        solver.parameters.random_seed = int(seed)
        journal_seconds = 0.
        if recovery_path is not None:
            jt = time.monotonic()
            path = Path(recovery_path)
            path.parent.mkdir(parents=True,exist_ok=True)
            data = dict(schema='cp_linear_recovery_v1',context=recovery_context,
                parent_genes={k:getattr(self.parent,k).tolist() for k in ('os','ma','agv_assign','agv_speed')},
                cluster=sorted(self.cluster),objective=self.objective,epsilon=self.epsilon,
                families=sorted(self.families),requested_solve_cap=seconds,seed=int(seed),
                parameters_before_deadline_adjustment=str(solver.parameters),
                deadline_monotonic=deadline,pid=os.getpid(),status='SOLVE_INPUT_PREPARED',
                model_variables=len(self.model.proto.variables),model_constraints=len(self.model.proto.constraints))
            temp = path.with_suffix('.tmp')
            with temp.open('w') as handle:
                json.dump(data,handle)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(path)
            journal_seconds = time.monotonic()-jt
        # Logging is charged. The persisted requested cap is an upper bound;
        # the actual cap also respects this run's remaining absolute deadline.
        if deadline is not None:
            seconds = min(seconds,max(0.,deadline-time.monotonic()-0.01))
            if seconds<=0.001:
                return None,dict(status='JOURNAL_EXCEEDED_DEADLINE_RESERVE',has_candidate=False,
                    accepted_epsilon=False,build_seconds=self.build_seconds,journal_seconds=journal_seconds,
                    solve_seconds=0.,solve_cpu_seconds=0.,total_seconds=time.monotonic()-self.started)
            solver.parameters.max_time_in_seconds = float(seconds)
        t,cpu = time.monotonic(),time.process_time()
        status = solver.solve(self.model)
        solve_seconds,solve_cpu = time.monotonic()-t,time.process_time()-cpu
        log = dict(status=solver.status_name(status),build_seconds=self.build_seconds,journal_seconds=journal_seconds,
            solve_seconds=solve_seconds,solve_cpu_seconds=solve_cpu,
            branches=int(solver.num_branches),conflicts=int(solver.num_conflicts),
            objective_index=self.objective,cluster=sorted(self.cluster),families=sorted(self.families),
            epsilon=self.epsilon,solver_workers=1,solve_cap_seconds=seconds,
            model_variables=len(self.model.proto.variables),model_constraints=len(self.model.proto.constraints),
            has_candidate=False,accepted_epsilon=False)
        if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):
            log['total_seconds']=time.monotonic()-self.started
            return None,log
        value = lambda x: int(x) if isinstance(x,int) else int(solver.value(x))
        child = self.parent.copy()
        outorder = []
        cp_trace = {}
        for p,attrs in enumerate(self.rows):
            f,mi,av,sv = [value(x) for x in attrs]
            child.os[p] = self.keys[f][0]
            child.ma[f],child.agv_assign[f],child.agv_speed[f] = mi,av,sv
            outorder.append(f)
            cp_trace[f] = [value(x) for x in self.trace[p]]
        child.invalidate()
        _,actualorder = flat_order(child)
        assert outorder == actualorder, 'Operation identities lost during OS round-trip'
        for p,f in enumerate(self.order):
            if f not in self.cluster:
                assert outorder[p]==f
                assert all(getattr(child,k)[f]==getattr(self.parent,k)[f]
                           for k in ('ma','agv_assign','agv_speed'))
        cp_obj = [value(x) for x in self.objectives]
        replay = integer_replay(child)
        assert cp_trace == replay['trace'], 'CP times differ from independent integer replay'
        assert cp_obj == replay['objectives'], 'CP objectives differ from integer replay'
        tf = time.monotonic()
        actual = child.objectives.to_array()
        schedule = child.schedule
        time_error = max(abs(float(v)/TIME_SCALE-getattr(schedule,field)[self.keys[f]])
            for f,times in cp_trace.items() for v,field in zip(times,
                ['transport_start','transport_end','op_start','op_end']))
        predicted = np.asarray(cp_obj)/np.array([TIME_SCALE,ENERGY_SCALE,TIME_SCALE])
        delta = predicted-actual
        assert time_error <= 2*self.n/TIME_SCALE+1e-8, 'Float time discrepancy exceeds rounding bound'
        assert abs(delta[2])<=self.n/TIME_SCALE+1e-8, 'Float workload discrepancy exceeds rounding bound'
        assert abs(delta[1])/max(1.,abs(actual[1]))<=1e-4, 'Float energy discrepancy exceeds gate'
        accepted = self.epsilon is None or all(v is None or actual[k]<=v+1e-8 for k,v in enumerate(self.epsilon))
        log.update(has_candidate=True,accepted_epsilon=accepted,integer_replay_exact=True,
            integer_objectives=cp_obj,predicted_objectives=predicted.tolist(),
            decoded_objectives=actual.tolist(),objective_errors=delta.tolist(),
            max_time_error=time_error,validation_seconds=time.monotonic()-tf,
            integer_best_bound=float(solver.best_objective_bound),
            changed_genes={k:int(np.count_nonzero(getattr(child,k)!=getattr(self.parent,k)))
                           for k in ('os','ma','agv_assign','agv_speed')},
            genes={k:getattr(child,k).tolist() for k in ('os','ma','agv_assign','agv_speed')},
            total_seconds=time.monotonic()-self.started)
        return (child if accepted else None),log
