"""Collect diagnostic clocks and render the current GDN profiling report.

Run through scripts/profile_gdn_mega.sh. Diagnostic code is generated only in
ignored benchmark_data; production sources and the production library stay intact.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SLOTS = 32
CUBE_BINS = {3: 'Startup', 4: 'Q/K × paired state', 5: 'Load diagonal inverses',
             6: 'Join 16 → 32', 7: 'Join 32 → 64', 8: 'Join 64 → 128',
             9: 'Load attention', 10: 'Wait for RHS', 11: 'Solve residual + exchange',
             12: 'Output/state products + exchange', 13: 'Wait for state feedback',
             14: 'Reload paired state'}
VECTOR_BINS = {3: 'First inverse + state setup', 4: 'Wait for state load',
               5: 'Next chunk weights', 6: 'Wait for projections', 7: 'Prepare RHS',
               8: 'Next diagonal inverses + Gamma', 9: 'Wait for output/state products',
               10: 'Finish output + update/pair state', 16: 'Final state-load acknowledgement'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_diagnostic(destination):
    destination.mkdir(parents=True, exist_ok=True)
    sources = {p.name: p.read_text() for p in (ROOT / 'csrc/kernel/gdn_mega').glob('*')
               if p.suffix in ('.hpp', '.cpp')}
    def replace(name, old, new):
        assert sources[name].count(old) == 1, (name, old, sources[name].count(old))
        sources[name] = sources[name].replace(old, new)
    # Stage clocks describe the H16 persistent schedule. Isolate that function
    # so similarly named synchronization in token producers is not instrumented.
    f = 'fused_pipeline.hpp'
    prefix, body = sources[f].split('template<int H,int C,int D,int V=D,bool Prepared=false>', 1)
    body, suffix = body.split('// K1: token/key-head parallel work', 1)
    sources[f] = 'template<int H,int C,int D,int V=D,bool Prepared=false>' + body
    replace(f, '__gm__ float* kk,__gm__ float* kk_upper,__gm__ float* qk,__gm__ float* lower,__gm__ float* full)\n{',
            '__gm__ float* kk,__gm__ float* kk_upper,__gm__ float* qk,__gm__ float* lower,__gm__ float* full,uint64_t* report)\n{')
    replace(f, '    int block=get_block_idx();', '    uint64_t stamp=get_sys_cnt();\n    int block=get_block_idx();')
    replace(f, '        int64_t head_work=work/(D/V);', '        stamp=get_sys_cnt();\n        int64_t head_work=work/(D/V);')
    replace(f, '        for(int64_t pos=start;pos<end;pos+=C){',
            '        report[3]+=get_sys_cnt()-stamp;\n        for(int64_t pos=start;pos<end;pos+=C){\n            ++report[2];')
    replace(f, '            wait_flag_dev(7);',
            '            stamp=get_sys_cnt();\n            wait_flag_dev(7);\n            report[4]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();')
    replace(f, '            wait_flag_dev(5);',
            '            report[5]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n            wait_flag_dev(5);\n            report[6]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();')
    replace(f, '                if(pos+C<end)mega_finish_inverse',
            '            report[7]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n                if(pos+C<end)mega_finish_inverse')
    replace(f, '            wait_flag_dev(9);',
            '            report[8]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n            wait_flag_dev(9);\n            report[9]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();')
    replace(f, '            mega_update_state<V>(exchange,alpha,final_state+head_work*D*D+lane*V,pos+C>=end);',
            '            mega_update_state<V>(exchange,alpha,final_state+head_work*D*D+lane*V,pos+C>=end);\n            report[10]+=get_sys_cnt()-stamp;')
    replace(f, '            Tail ktail(valid),qtail(valid);',
            '            stamp=get_sys_cnt();\n            Tail ktail(valid),qtail(valid);')
    replace(f, '            if constexpr(Prepared)ffts_cross_core_sync',
            '            report[4]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n            if constexpr(Prepared)ffts_cross_core_sync')
    replace(f, '            mega_float_inverse<C>(inverse,solve_scratch);',
            '            report[5]+=get_sys_cnt()-stamp;\n            mega_float_inverse<C>(inverse,solve_scratch,nullptr,C,report);\n            stamp=get_sys_cnt();')
    replace(f, '            wait_flag_dev(6);\n            GlobalTensor<float',
            '            report[9]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n            wait_flag_dev(6);\n            report[10]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n            GlobalTensor<float')
    replace(f, '            TLOAD(native_residual,ex);pipe_barrier(PIPE_ALL);',
            '            TLOAD(native_residual,ex);pipe_barrier(PIPE_ALL);\n            report[11]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();')
    replace(f, '            wait_flag_dev(6);mega_load_pair<D,V>(state,sh,sh+V,D,2*V);',
            '            report[12]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n            wait_flag_dev(6);\n            report[13]+=get_sys_cnt()-stamp; stamp=get_sys_cnt();\n            mega_load_pair<D,V>(state,sh,sh+V,D,2*V);\n            report[14]+=get_sys_cnt()-stamp;')
    replace(f, '        wait_flag_dev(7);\n#endif',
            '        stamp=get_sys_cnt(); wait_flag_dev(7); report[16]+=get_sys_cnt()-stamp;\n#endif')
    sources[f] = prefix + sources[f] + '// K1: token/key-head parallel work' + suffix
    f = 'recurrence_pipeline.hpp'
    replace(f, 'mega_float_inverse(Mat& inverse,__gm__ float* scratch,__gm__ float* prepared=nullptr,int valid=C)',
            'mega_float_inverse(Mat& inverse,__gm__ float* scratch,__gm__ float* prepared=nullptr,int valid=C,uint64_t* report=nullptr)')
    for level, slot in [(16, 6), (32, 7), (64, 8)]:
        args = f'{level},C' + (',true' if level == 64 else '')
        call = f'mega_float_join_level<{args}>(inverse,scratch,prepared,valid);'
        replace(f, f'    {call}',
                f'    {{ uint64_t stamp=get_sys_cnt(); {call} report[{slot}]+=get_sys_cnt()-stamp; }}')
    f = 'kernel_gdn_mega.cpp'
    # This diagnostic is intentionally H16-only. Keep the normal arithmetic
    # unchanged, and omit the unused H8 dispatch from the instrumented copy.
    start = sources[f].index('    // Two 64-column workers per head')
    end = sources[f].index('    mega_shared_grams<H, C, D>(', start)
    sources[f] = sources[f][:start] + sources[f][end:]
    replace(f, '    // Phase 1: vector gate scan/layout overlaps cube KK/QK products.',
            '    uint64_t report[32]={0}; uint64_t phase_stamp=get_sys_cnt();\n    // Phase 1: vector gate scan/layout overlaps cube KK/QK products.')
    replace(f, '    mega_sync_all();',
            '    mega_sync_all();\n    report[0]=get_sys_cnt()-phase_stamp; phase_stamp=get_sys_cnt();')
    replace(f, 'reinterpret_cast<__gm__ float*>(msk_full_ptr));',
            'reinterpret_cast<__gm__ float*>(msk_full_ptr),report);')
    replace(f, '    pipe_barrier(PIPE_ALL);\n}', '''    pipe_barrier(PIPE_ALL);
    report[1]=get_sys_cnt()-phase_stamp;
    mega_sync_all(); // Prefix workspace is dead after every reader completes.
#if defined(__DAV_CUBE__)
    int unit=0;
#else
    int unit=1+get_subblockid();
#endif
    auto data=reinterpret_cast<__gm__ uint64_t*>(g_sum_ptr)+(get_block_idx()*3+unit)*32;
    for(int i=0;i<32;++i)data[i]=report[i];
    for(int i=0;i<32;i+=8)dcci(data+i,SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}''')
    for name, text in sources.items():
        (destination / name).write_text(text)
    (destination / 'source_manifest.json').write_text(json.dumps({
        'production': {p.name: sha(p) for p in (ROOT / 'csrc/kernel/gdn_mega').glob('*') if p.suffix in ('.hpp', '.cpp')},
        'instrumented': {name: sha(destination / name) for name in sources},
        'generator_sha256': sha(__file__),
    }, indent=2) + '\n')
    print(f'Prepared isolated diagnostic at {destination}')


def measure_diagnostic(output):
    import ctypes
    import torch
    import torch_npu  # noqa: F401
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_gdn_mega import _make_inputs, _prepare_gdn_mega
    from gdn_mega_validation import recurrent_reference, error_metrics, POLICY
    from pto_kernels import pto_gdn_mega
    torch.set_num_threads(4)
    torch.npu.set_device('npu:0')
    torch.npu.config.allow_internal_format = False
    libraries = {s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
                 if s.split()[-1].endswith('/libno_workspace_kernel.so')}
    assert len(libraries) == 1
    library = Path(libraries.pop())
    assert sha(library) != sha(ROOT / 'build/lib/libno_workspace_kernel.so'), 'Diagnostic library was not loaded'
    fn = ctypes.CDLL('libascend_hal.so').halGetDeviceInfo
    fn.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    frequency = ctypes.c_longlong()
    assert fn(0, 0, 25, ctypes.byref(frequency)) == 0
    hz = frequency.value * 1000
    inputs = _make_inputs(4096, 16, 4)
    initial = torch.randn(1, 16, 128, 128, dtype=torch.float16) * .1
    golden = recurrent_reference(*inputs, initial)
    launch = _prepare_gdn_mega(*(x.to('npu:0') for x in inputs),
                              torch.tensor([0, 4096], dtype=torch.int32, device='npu:0'), initial.to('npu:0'))
    assert launch['g_sum'].numel()*4 >= 24*3*SLOTS*8
    ticks, times, checks = [], [], []
    for iteration in range(18):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record(); pto_gdn_mega(**launch); end.record(); end.synchronize()
        if iteration in (0, 17):
            metrics = {key: error_metrics(launch[field].cpu(), expected.reshape_as(launch[field])) for key, field, expected in
                       [('raw_output', 'out', golden[0]), ('final_state', 'final_state', golden[1])]}
            assert all(m['passed'] for m in metrics.values()), metrics
            checks.append(metrics)
        if iteration >= 3:
            ticks.append(launch['g_sum'].view(torch.int64).flatten()[:24*3*SLOTS].reshape(24, 3, SLOTS).cpu().clone())
            times.append(start.elapsed_time(end)*1000)
    ticks = torch.stack(ticks)
    assert (ticks[:, :16, :, 2] == 32).all() and (ticks[:, 16:, :, 2] == 0).all()
    for unit, slots in ((0, CUBE_BINS), (1, VECTOR_BINS), (2, VECTOR_BINS)):
        assert (ticks[:, :, unit, list(slots)].sum(-1) <= ticks[:, :, unit, 1]).all(), 'Overlapping diagnostic scopes'
    medians = ticks.double().median(0).values * 1e6 / hz
    result = dict(shape={'heads':16,'key_heads':4,'length':4096,'initial_state':True},
                  clock_hz=hz, warmups=3, samples=15, raw_ticks=ticks.tolist(),
                  per_core_median_us=medians.tolist(), event_us=times,
                  event_median_us=statistics.median(times), policy=POLICY, numerical_checks=checks,
                  library_sha256=sha(library), production_library_sha256=sha(ROOT/'build/lib/libno_workspace_kernel.so'),
                  source_manifest=json.loads((library.parent/'source_manifest.json').read_text()),
                  bins={'cube':CUBE_BINS,'vector':VECTOR_BINS},
                  caveat='Instrumented scope clocks include waits/drains and perturb scheduling; units overlap. Not production latency.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Diagnostic passed FP64 elementwise checks; median event {result["event_median_us"]:.3f} us')


def checked_log(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{"length"')]
    assert len(records) == 1, f'Missing or duplicate accuracy record: {path}'
    result = records[0]
    assert all(m['passed'] for m in result['numerical_errors'].values()), path
    return result


def collect_hardware(root, h, hg, length):
    directory = root / f'device_h{h}_t{length}'
    runs = list(directory.glob('OPPROF_*'))
    assert len(runs) == 1, directory
    run = runs[0]
    tables = {p.stem: list(csv.DictReader(p.open())) for p in run.glob('*.csv')}
    task_us = float(tables['OpBasicInfo'][0]['Task Duration(us)'])
    active = min(h, 24)
    medians = {}
    for unit in ('cube0', 'vector0', 'vector1'):
        values = {}
        for rows in tables.values():
            if not rows or 'block_id' not in rows[0]: continue
            selected = [r for r in rows if int(r['block_id']) < active and r['sub_block_id'] == unit]
            for key in rows[0]:
                if not key or key in ('block_id', 'sub_block_id'): continue
                try: numbers = [float(r[key]) for r in selected]
                except (ValueError, TypeError): continue
                if numbers: values[key] = statistics.median(numbers)
        medians[unit] = values
    return dict(heads=h, key_heads=hg, length=length, task_us=task_us,
                active_groups=active, heads_per_group=h/active, folder=str(run),
                accuracy=checked_log(root / (directory.name+'.log')),
                active_group_medians=medians, tables=tables,
                csv_sha256={p.name:sha(p) for p in run.glob('*.csv')})


def save_plot(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
    # Matplotlib emits trailing whitespace in generated SVGs.
    svg = stem.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def render_report(root):
    from dataclasses import asdict
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    traceviz = ROOT.parent / 'traceviz-mcp'
    sys.path.insert(0, str(traceviz))
    from traceviz.loader import load_trace, read_kernel_name
    from traceviz.analyse import analyse, metric_overlap
    from traceviz.render import render_timeline
    from traceviz.report import format_summary, format_overlap
    out = ROOT / 'docs/gdn_mega'
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    runs = [collect_hardware(root, *shape) for shape in
            [(16,4,4096), (24,8,4096), (48,16,4096), (16,4,32768)]]
    diagnostic = json.loads((root/'diagnostic.json').read_text())
    diagnostic_runs = list((root/'device_diagnostic').glob('OPPROF_*'))
    assert len(diagnostic_runs) == 1
    diagnostic['hardware_tables'] = {p.stem:list(csv.DictReader(p.open())) for p in diagnostic_runs[0].glob('*.csv')}
    diagnostic['profiled_task_us'] = float(diagnostic['hardware_tables']['OpBasicInfo'][0]['Task Duration(us)'])
    diagnostic['profile_accuracy'] = checked_log(root/'device_diagnostic.log')
    assert diagnostic['production_library_sha256'] == sha(ROOT/'build/lib/libno_workspace_kernel.so')
    assert all(sha(ROOT/'csrc/kernel/gdn_mega'/name) == value
               for name, value in diagnostic['source_manifest']['production'].items())
    assert all(m['passed'] for check in diagnostic['numerical_checks'] for m in check.values())
    assert all(run['accuracy']['validation_policy'] == diagnostic['policy'] for run in runs)
    sim = list((root/'simulator_h16_t512').glob('OPPROF_*'))
    assert len(sim) == 1
    sim = sim[0]
    sim_accuracy = checked_log(root/'simulator_h16_t512.log')
    assert sim_accuracy['validation_policy'] == diagnostic['policy']
    print('Loading current production simulator trace through traceviz...', flush=True)
    td = load_trace(sim)
    active_units = [f'core{i}.{kind}' for i in range(16) for kind in ('cubecore0','veccore0','veccore1')]
    stats = analyse(td, cores=active_units, label=sim.name, kernel=read_kernel_name(sim))
    selected = ['core0.cubecore0','core0.veccore0','core0.veccore1']
    engines = [e for e in ('SCALAR','FLOWCTRL','CUBE','VECTOR','MTE1','MTE2','MTE3','FIXPIPE','FIXP') if e in td.engines]
    render_timeline(td, str(figures/'profile_timeline.png'), units=selected, engines=engines,
                    width_px=1800, row_px=29, title='Production simulator: one cube + two vector subcores, H16/Hg4, 512 tokens')
    window = (td.t_min + td.span()*.5, td.t_max)
    render_timeline(td, str(figures/'profile_zoom.png'), units=selected, engines=engines,
                    window=window, width_px=1800, row_px=29,
                    title='Production simulator: second half of execution; gaps and transfer/compute overlap')
    render_timeline(td, str(figures/'profile_core_timeline.png'),
                    units=[f'core{i}.cubecore0' for i in (0,7,15,16,23)], engines=engines,
                    aggregate='unit', width_px=1800, row_px=38,
                    title='Production simulator: persistent groups 0–15 versus setup-only groups 16–23')
    overlaps = []
    fix = next(e for e in ('FIXPIPE','FIXP') if e in td.engines)
    for a,b,cores in [(fix,'CUBE',['core0.cubecore0']), ('MTE2','CUBE',['core0.cubecore0']),
                      ('MTE3','VECTOR',['core0.veccore0','core0.veccore1'])]:
        overlaps.append(metric_overlap(td,a,b,cores=cores,window=window))
    (out/'data/profile_traceviz.txt').write_text(format_summary(stats)+'\n\n'+'\n\n'.join(format_overlap(v) for v in overlaps)+'\n')

    # Hardware counters are independent bars: overlapping engines cannot be stacked.
    labels = [('cube0','aic_cube_time(us)','Cube matrix arithmetic'),
              ('cube0','aic_mte1_time(us)','Cube MTE1: L1 → operands'),
              ('cube0','aic_mte2_time(us)','Cube MTE2: GM/L2 → L1'),
              ('cube0','aic_fixpipe_time(us)','Cube FIXPIPE: results → GM/L2'),
              ('cube0','aic_scalar_time(us)','Cube scalar/control'),
              ('vector0','aiv_vec_time(us)','Vector arithmetic'),
              ('vector0','aiv_mte2_time(us)','Vector MTE2: GM/L2 → UB'),
              ('vector0','aiv_mte3_time(us)','Vector MTE3: UB → GM/L2'),
              ('vector0','aiv_scalar_time(us)','Vector scalar/control')]
    fig, ax = plt.subplots(figsize=(12,6))
    y = np.arange(len(labels))
    for run,shift,color in [(runs[0],-.17,'#2369bd'),(runs[3],.17,'#e08b28')]:
        values=[100*run['active_group_medians'][u][k]/run['task_us'] for u,k,_ in labels]
        bars=ax.barh(y+shift,values,height=.32,color=color,label=f"T={run['length']:,}; task {run['task_us']:.1f} µs")
        ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    ax.set_yticks(y,[x[2] for x in labels]);ax.invert_yaxis();ax.set_xlim(0,65)
    ax.set_xlabel('Median active-group counter time / profiled task duration (%)')
    ax.set_title('Real hardware: compute is intermittent, at both 4K and 32K\nIndependent counters overlap; do not add these bars')
    ax.legend();ax.grid(axis='x',alpha=.2);fig.tight_layout()
    save_plot(fig,figures/'profile_hardware');plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,4))
    for run,shift,color in [(runs[0],-.18,'#2369bd'),(runs[1],.18,'#20a083')]:
        rows=[r for r in run['tables']['PipeUtilization'] if r['sub_block_id']=='cube0']
        values=[float(r['aic_time(us)']) for r in sorted(rows,key=lambda r:int(r['block_id']))]
        ax.bar(np.arange(24)+shift,values,width=.35,color=color,label=f"H/Hg={run['heads']}/{run['key_heads']}")
    ax.set_xticks(range(24));ax.set_xlabel('Physical cube/vector group');ax.set_ylabel('Cube-core task span (µs)')
    ax.set_title('Real hardware, 4K: 16 heads leave 8 persistent groups unused\nTask span includes waiting; it is not arithmetic busy time')
    ax.legend();fig.tight_layout();save_plot(fig,figures/'profile_groups');plt.close(fig)

    clock=np.array(diagnostic['per_core_median_us'])
    bins={kind:{slot:float(np.median(clock[:16,unit,slot])) for slot in mapping}
          for kind,unit,mapping in [('cube',0,CUBE_BINS),('vector',1,VECTOR_BINS)]}
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    for ax,kind,mapping in zip(axes,('cube','vector'),(CUBE_BINS,VECTOR_BINS)):
        values=[bins[kind][s] for s in mapping]
        colors=['#d88531' if 'Wait' in text else '#2369bd' if kind=='cube' else '#20a083' for text in mapping.values()]
        bars=ax.barh(np.arange(len(values)),values,color=colors)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set_yticks(np.arange(len(values)),list(mapping.values()),fontsize=9);ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.22);ax.set_xlabel('Elapsed scope µs, summed across 32 chunks')
        ax.set_title('Cube + its scalar controller' if kind=='cube' else 'Vector subcore 0 + its scalar controller')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Isolated diagnostic, H16/Hg4, 4K: scope clocks include compute, transfers and waits\nThe two sides run concurrently; do not add cube and vector times',fontsize=13)
    fig.tight_layout();save_plot(fig,figures/'profile_stages');plt.close(fig)

    # Conceptual dependency graph, deliberately not presented as measured timing.
    fig,ax=plt.subplots(figsize=(15,8));ax.set_xlim(0,15);ax.set_ylim(0,8);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',facecolor=color,edgecolor='#34495e'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#34495e',linewidth=1.5))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=9)
    ax.text(7.5,7.8,'One launch, two global phases — dependency schematic (not a timing diagram)',ha='center',fontsize=15,weight='bold')
    box(.2,6.35,4.0,.95,'Phase 1 · cube\nShared KKᵀ and QKᵀ per key head','#dce9fb')
    box(5.0,6.35,4.0,.95,'Phase 1 · vector\nCompensated gate prefixes / β layout','#ddf3e9')
    box(10.0,6.35,3.8,.95,'Global barrier\nThen one persistent worker per head','#eeeeee')
    ax.plot([2.2,2.2,11.9],[7.4,7.55,7.55],color='#34495e',lw=1.5)
    arrow((11.9,7.55),(11.9,7.4));arrow((9.1,6.8),(9.9,6.8))
    ax.text(.2,5.9,'Phase 2: each 128-token chunk; cube and vector lanes cooperate',fontsize=12,weight='bold')
    box(.2,4.45,2.6,.95,'Cube: Q/K × state\nFP16 high + low','#dce9fb')
    box(3.35,4.45,2.6,.95,'Cube: FP32 joins\n16 → 32 → 64 → 128','#dce9fb')
    box(6.5,4.45,2.6,.95,'Cube: residual solve\nR = inverse × RHS','#dce9fb')
    box(10.0,4.45,3.8,.95,'Cube: attention × R\nand Γᵀ × R, stacked','#dce9fb')
    for a,b in [((2.9,4.92),(3.25,4.92)),((6.05,4.92),(6.4,4.92)),((9.2,4.92),(9.9,4.92))]:arrow(a,b)
    box(.2,2.35,2.6,1.0,'Vector: next weights\nβ, exp(Gᵢ−Gⱼ), masks','#ddf3e9')
    box(3.35,2.35,2.6,1.0,'Vector: current RHS\nβ(V − exp(G) K S)\nretain output base','#ddf3e9')
    box(6.5,2.35,2.6,1.0,'Vector: next inverse\n16×16 diagonals\nand Γ weighting','#ddf3e9')
    box(10.0,2.35,3.8,1.0,'Vector: finish output, update S\nFP32 S stays in UB\nexport high + low for next chunk','#ddf3e9')
    for a,b in [((2.9,2.85),(3.25,2.85)),((6.05,2.85),(6.4,2.85)),((9.2,2.85),(9.9,2.85))]:arrow(a,b)
    arrow((1.5,4.35),(4.5,3.45),'projections / flag 5')
    arrow((5.6,3.45),(7.4,4.35),'RHS / flag 6')
    arrow((12.0,4.35),(12.0,3.45),'results / flag 9')
    ax.plot([11,11,.05,.05],[2.2,1.0,1.0,4.7],color='#ad5d16',lw=1.5)
    arrow((.05,4.7),(.15,4.7))
    ax.text(6.5,1.25,'State feedback gates the next chunk (flag 6, then reload and flag 7)',ha='center',color='#ad5d16',fontsize=11)
    ax.text(7.5,.35,'Cross-engine payloads use GM-addressed scratch, often served by L2.\nThe scalar controllers issue instructions and wait on flags; MTE/FIXPIPE move data independently.',ha='center',fontsize=11)
    save_plot(fig,figures/'profile_pipeline');plt.close(fig)

    report=dict(raw_directory=str(root),source_commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT}','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
                kernel_commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT}','log','-1','--format=%H','--',':(glob)csrc/kernel/gdn_mega/*.cpp',':(glob)csrc/kernel/gdn_mega/*.hpp'],text=True,cwd=ROOT).strip(),
                production_library_sha256=sha(ROOT/'build/lib/libno_workspace_kernel.so'),
                traceviz_commit=subprocess.check_output(['git','-c',f'safe.directory={traceviz}','rev-parse','HEAD'],text=True,cwd=traceviz).strip(),
                hardware=runs,diagnostic=diagnostic,diagnostic_median_bins_us=bins,
                simulator=dict(folder=str(sim),trace_sha256=sha(sim/'simulator/trace.json'),accuracy=sim_accuracy,
                               span_ns=td.span(),events=td.n,summary=asdict(stats),overlap_window_ns=window,overlaps=overlaps),
                scripts={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/profile_gdn_mega.sh']})
    stamp = Path(runs[0]['folder']).name.split('_')[1]
    report['profiled_date'] = f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}'
    (out/'data/profile.json').write_text(json.dumps(report,indent=2)+'\n')
    write_profile_markdown(report,out/'profile.md')
    print(f'Saved profile.md, profile.json and figures under {out}')



def write_profile_markdown(report, path):
    r = report['hardware'][0]
    d = report['diagnostic']
    cube = report['diagnostic_median_bins_us']['cube']
    vector = report['diagnostic_median_bins_us']['vector']
    # Numeric keys are integers before JSON serialization.
    phase1 = statistics.median(row[0][0] for row in d['per_core_median_us'][:16])
    phase2 = statistics.median(row[0][1] for row in d['per_core_median_us'][:16])
    joins = sum(cube[k] for k in (6,7,8))
    heatmap = json.loads((path.parent/'data/heatmap.json').read_text())
    assert heatmap['library_sha256']['current'] == report['production_library_sha256']
    graph = {(x['heads'],x['key_heads'],x['length']):x['current_ms']*1000 for x in heatmap['records']}
    rows = '\n'.join(f"| {x['heads']}/{x['key_heads']} | {x['length']:,} | {graph[x['heads'],x['key_heads'],x['length']]:.2f} | {x['task_us']:.2f} | {x['active_groups']}/24 | {x['heads_per_group']:g} |" for x in report['hardware'])
    stage_rows = '\n'.join(f'| {name} | {cube[k]:.2f} | {cube[k]/32:.2f} |' for k,name in CUBE_BINS.items())
    vector_rows = '\n'.join(f'| {name} | {vector[k]:.2f} |' for k,name in VECTOR_BINS.items())
    metrics = r['active_group_medians']
    prep = vector[5]+vector[8]
    duration = r['task_us']
    overhead = 100*(d['profiled_task_us']/duration-1)
    overlaps = '\n'.join(f"| {o['metric_a']} against {o['metric_b']} | {o['overlap_pct_median']:.1f}% | {o['exposed_ns_median']/1000:.2f} |" for o in report['simulator']['overlaps'])
    # Dense matrix-work counts, not the profiler's instruction-count proxy.
    chunks=4096//128; heads=16
    products=[('Shared KKᵀ + QKᵀ',4*128**3*4*chunks,phase1,'FP16 → FP32'),
              ('Q/K × paired state',8*128**3*heads*chunks,cube[4],'FP16 high + low → FP32'),
              ('Inverse joins',sum(4*b**3*(128//(2*b)) for b in (16,32,64))*heads*chunks,joins,'FP32'),
              ('Apply inverse to RHS',2*128**3*heads*chunks,cube[11],'FP32'),
              ('Attention and state products',4*128**3*heads*chunks,cube[12],'FP32')]
    flop_rows='\n'.join(f'| {name} | {flops/1e9:.3f} | {us:.2f} | {flops/us/1e6:.2f} | {precision} |' for name,flops,us,precision in products)
    text=f'''# Current GDN kernel: profile and breakthrough opportunities

Profiled **{report['profiled_date']}**, production kernel **{report['kernel_commit'][:7]}**, library
`{report['production_library_sha256']}` on Ascend 910B2.
This is a measurement and analysis update; no production arithmetic, scheduling,
interface or allocation changed. [Speed heatmap](heatmap.md) · [Accuracy regimes](error_heatmap.md).

**The kernel is limited by its chunk dependency chain and data handoffs, with
additional unused core capacity at small head counts. It is not running anywhere
near continuous cube arithmetic.** At H16/Hg4/T4096, the hardware task takes
**{duration:.2f} µs** and the active-group median cube arithmetic time is
**{metrics['cube0']['aic_cube_time(us)']:.2f} µs ({100*metrics['cube0']['aic_cube_time(us)']/duration:.1f}%)**.
The same pattern persists at 32K. A general 2× gain needs a structural change to
solve/transfer scheduling or the recurrent state dependency; saving a few scalar
instructions or more shared Gram work cannot provide it by itself.

## What was measured

| H/Hg | Tokens | Production graph median µs | Fresh hardware profiler task µs | Persistent groups | Heads per active group |
|---|---:|---:|---:|---:|---:|
{rows}

The graph medians are the separately recorded current-kernel heatmap run, with
15 trials × 20 calls. The profiler tasks are fresh `msprof op --aic-metrics=Roofline`
runs on the same production library; profiling replays/counters are not an
additional set of graph medians. Inputs are the ordinary heatmap distribution:
normalized Gaussian Q/K, Gaussian V, uniform β in [0,1), log-sigmoid Gaussian g,
nonzero supplied state, seed 0, D=C=128. This is not a worst-case input audit.
There is no data-dependent fallback in this kernel.

Four production hardware profiles, the 512-token simulator launch, and the
instrumented diagnostic's first/last launches and profiler launch all pass
independent CPU FP64 checks on every output/final-state element:
`abs(error) <= 1e-4 + 1e-3*abs(reference)`, finiteness, and relative RMSE < 0.005.
No tolerances were adjusted. This profiling run does not replace the complete
245-test regression suite previously recorded in [the overview](README.md).

Three evidence sources are kept distinct throughout this report:

- **Hardware counters:** unmodified production binary, real device. Pipe times
  overlap; percentages below use task duration as their denominator.
- **Stage clocks:** a separately compiled diagnostic copy with `get_sys_cnt()`.
  It has 3 warmups and 15 samples; tables use per-core sample medians, then the
  median over groups 0–15. Clock frequency is {d['clock_hz']:,} Hz, queried from HAL.
  The instrumented profiler task is {d['profiled_task_us']:.2f} µs
  ({overhead:+.1f}% versus production), so stage attribution is approximate.
  Single-launch event median {d['event_median_us']:.2f} µs also includes submission/event
  effects and is not substituted for graph latency. Scope clocks include
  transfers, instruction issue and waits. Medians do not add exactly.
- **Traceviz timeline:** unmodified production binary in the simulator,
  **512 tokens / four chunks**, {report['simulator']['events']:,} events,
  span {report['simulator']['span_ns']/1000:.2f} µs. It explains ordering and overlap,
  not the absolute latency or critical-path proportions of the 4K hardware run.

## Hardware vocabulary and the two global phases

There are **24 hardware groups**, each with one cube core and two vector
subcores. Both core types have scalar control/instruction machinery. “SCALAR”
in a trace is that core's controller, not a third pool of independent workers.

| Lane / memory | Role in this kernel |
|---|---|
| CUBE | Matrix multiply/accumulate. FP16 inputs with FP32 accumulation for grams/projections; native FP32 operands for joins, residual and output/state products. |
| VECTOR | Gate weighting, masks, 16×16 diagonal substitution, RHS subtraction, output addition, authoritative FP32 state update and high/low encoding. |
| SCALAR | Loop/index/address work, tile setup, instruction issue and synchronization control. Scalar busy time is distinct from time waiting on flags. |
| FLOWCTRL | Simulator control-flow lane; includes `WAIT_FLAG_DEVI` as well as branching. A busy control lane can mean waiting, not useful math. |
| MTE1 | Move operands from L1 into cube operand buffers L0A/L0B. |
| MTE2 | Load GM-addressed data into L1 (cube) or UB (vector); L2 often serves these requests. |
| MTE3 | Export vector UB results to GM-addressed scratch/output. |
| FIXP / FIXPIPE | Export cube accumulator L0C results, including FP32 scratch stores. |
| L1 / UB | Cube-local operand storage / vector-local storage. The authoritative FP32 state stays in vector UB across chunks. |

![Dependency diagram](figures/profile_pipeline.png)

**Phase 1:** gate scan/layout and shared KKᵀ/QKᵀ execute concurrently, then a
global barrier joins them. **Phase 2:** persistent head workers loop over
128-token chunks, retaining state locally. Suboperations in the diagram are
not separate launches or new global stages. The diagnostic's median cube-side
phase clocks are {phase1:.2f} µs for setup and {phase2:.2f} µs for the persistent phase.

The inverse still has a full **128×128** domain. Vector solves eight 16×16
diagonal blocks; cube joins them at widths 16, 32 and 64 to obtain the full
inverse. There is no power-series truncation or repeated squaring. The
16×16 dimension is an internal solve tile, not a smaller token chunk.
The diagram's “next inverse” belongs to chunk t+1; the cube joins consume the
inverse blocks prepared for chunk t. Flag 5 acknowledges that the cube has
loaded the current blocks, allowing the vector side to reuse their scratch.

## What the hardware is actually busy doing

![Hardware resource counters](figures/profile_hardware.png)

At H16/T4096, active-group median cube MTE2 time is
{metrics['cube0']['aic_mte2_time(us)']:.2f} µs and FIXPIPE time is
{metrics['cube0']['aic_fixpipe_time(us)']:.2f} µs. Vector arithmetic takes
{metrics['vector0']['aiv_vec_time(us)']:.2f} µs; vector MTE2 takes
{metrics['vector0']['aiv_mte2_time(us)']:.2f} µs. These are overlapping engine
counters, not a partition of the kernel duration.

Cube-side L2 read-hit rate is {metrics['cube0']['aic_read_hit_rate(%)']:.2f}%
and vector-side rate is {metrics['vector0']['aiv_read_hit_rate(%)']:.2f}%
(active-group medians). GM→L1 traffic is {metrics['cube0']['GM_to_L1_datas(KB)']/1024:.2f} MiB
per active cube group, and L0C→GM is {metrics['cube0']['L0C_to_GM_datas(KB)']/1024:.2f} MiB.
**GM-addressed traffic is not equivalent to HBM traffic.** These results support
transfer/dependency latency as a target; they do not prove that memory bandwidth
is saturated or that eliminating every byte saves its full counter time.

![Hardware core-group spans](figures/profile_groups.png)

With 16 heads, only 16/24 groups execute the persistent recurrence. With 24
heads all groups work; 48 heads run two heads per group. The near-doubling from
24 to 48 heads shows the cost of the additional head wave. Occupancy is therefore
a real opportunity for small H, but it cannot explain all inefficiency: 24 and
48 heads also leave large gaps between cube arithmetic operations.

## Where time goes inside the persistent loop

![Measured stage scopes](figures/profile_stages.png)

Cube-side diagnostic, H16/Hg4/T4096, 32 chunks per head. The final column
amortizes startup across those 32 chunks; it is not a claim that startup repeats.

| Cube scope | Per-head total µs | Amortized µs/chunk |
|---|---:|---:|
{stage_rows}

| Vector subcore 0 scope | Per-head total µs |
|---|---:|
{vector_rows}

The two vector subcores have the same responsibilities on different row halves;
all their samples are retained in the data. Their streams and the cube stream
run concurrently. **Do not add these two tables to estimate kernel time.**

Three consequences matter:

1. **The earlier RHS dependency is already hidden.** Its cube wait is only
   {cube[10]:.2f} µs over 32 chunks. Vector can form RHS while cube assembles the
   inverse. Optimizing that wait alone has essentially no available gain.
2. **The later state-feedback dependency is still exposed.** Cube waits
   {cube[13]:.2f} µs, then spends {cube[14]:.2f} µs reloading the paired state.
   Vector finalization itself takes {vector[10]:.2f} µs and includes output work
   before state update. Some next-chunk preparation also sits before that update.
   Which part can be hidden requires a changed schedule, not simply removing waits.
3. **Preparation and inverse transfers are substantial.** Vector next-chunk
   weighting plus diagonal/Gamma preparation totals {prep:.2f} µs, overlapping
   cube work. Cube joins alone total {joins:.2f} µs despite having little arithmetic.
   The final 64→128 join takes {cube[8]:.2f} µs; optimizing just that join cannot
   recover the entire inverse/feedback cost.

## Why the matrix-heavy parts have low effective TFLOPS

The following counts come from actual dense matrix dimensions, counting multiply
plus add as two FLOPs. They include both FP16 state components and all executed
full tiles, exclude vector operations, and do not use the profiler's
instruction-derived `aic_cube_fops` field as a general mixed-precision FLOP count.
Each row's FLOPs cover **all heads**, while its time is the representative elapsed
scope of those concurrently executing heads. These are effective device
throughputs **including transfers/waits**, not isolated arithmetic peak rates.
The shared-Gram time includes gate-scan overlap and the setup barrier.

| Work at H16/Hg4/T4096 | Matrix GFLOPs | Diagnostic elapsed µs | Effective TFLOP/s | Operand precision |
|---|---:|---:|---:|---|
{flop_rows}

Inverse joins are only about **4.5% of persistent matrix FLOPs**, yet occupy a
large elapsed scope. Each level executes two dependent products, exporting the
first result and reloading it before the second. Across levels there are
**six synchronized export/reload phases per chunk**, containing **14 matrix
product results** (four pairs at 16, two at 32, one at 64). Independent pairs
are already batched; the serial per-pair version is not the current algorithm.

The hardware's direct accumulator-to-L1 path narrows these results to FP16/BF16,
so retaining native FP32 currently takes a GM/L2 round trip. Small 16/32 products,
operand setup, synchronization, and those round trips dominate their low effective
TFLOPS. FP16 and FP32 arithmetic also have different throughput: their TFLOPS
should not be compared as if they had the same hardware ceiling.

L1 allocation reaches its 512 KiB boundary in the current schedule (the last
64 KiB holds residuals). Some allocations reuse earlier lifetimes. A proposed
extra buffer must fit those lifetimes; “double-buffer everything” is not feasible.

## Traceviz: see the gaps and the handoffs

![Full four-chunk simulator timeline](figures/profile_timeline.png)

This is one physical group: one cube core and both vector subcores. Read CUBE
and VECTOR as arithmetic, MTE/FIXP as data movement, and SCALAR/FLOWCTRL as the
controllers issuing and waiting for work. White regions mean no events on that
lane; they do not by themselves identify which dependency caused the gap.
Traceviz's aggregate occupancy includes wait instructions: a nearly full
aggregate lane does not mean the arithmetic units are nearly fully utilized.

![Zoomed simulator timeline](figures/profile_zoom.png)

In the **second half of this simulator trace**, traceviz measures overlap on the
same core, never by matching one core's stores against another core's computation:

| Movement pipe versus arithmetic | Movement time overlapped with arithmetic | Median exposed movement µs/core |
|---|---:|---:|
{overlaps}

“Exposed” is relative to the named arithmetic pipe only. It is not an additive
critical-path budget or a promised speedup; other work can overlap that movement.
The simulator has different absolute timing from hardware. Its ordering evidence
supports, but does not replace, the real-device stage/counter measurements.

![Active and inactive cube groups](figures/profile_core_timeline.png)

The compact trace confirms that groups 16–23 stop after setup for H16. The
hardware group-span chart above establishes the same behavior on the real device.

## Ranked opportunities for larger gains

These are hypotheses to test, not implemented speedups. The constraints are the
approved elementwise budget, all 160 numerical regimes in ordinary pytest, and
reasonable gate/correlated-key performance without a sequential fallback.

| Priority | Structural experiment | Evidence / possible benefit | Main tradeoff and decisive test |
|---|---|---|---|
| 1 | **Shorten the state feedback path.** Publish updated state before output finalization; reorganize output work and paired-state transfer so next projections can begin earlier. | About {cube[13]:.0f} µs of cube feedback wait plus {cube[14]:.0f} µs reload per 4K head. Vector currently finishes output before updating state. | Output base and pending inverse coefficients share UB lifetimes. Reordering alone may fail if it displaces other preparation. Measure feedback wait and whole-kernel latency together; preserve FP32 state. |
| 2 | **Avoid materializing the full inverse.** Apply a blocked triangular solve directly to RHS, keeping solved residual panels local and batching updates. | Could remove inverse join materialization and reduce dense inverse/application arithmetic. A straightforward 16-row block solve needs about 2.36M matrix FLOPs/chunk versus 5.57M for joins plus dense inverse application. | Eight dependent block steps and smaller matmuls can be slower despite fewer FLOPs. A previous partial factored-64 experiment was slower; a new version must remove transfers, not merely rearrange products. Test persistent reuse and transfer count before broad tuning. |
| 3 | **Use bounded helper workers for state-independent preparation.** Let otherwise idle groups prepare future weighted coefficients/inverses while head workers advance states. | H16 leaves eight groups unused; next-chunk vector preparation and inverse joins are substantial. | Requires bounded buffering, explicit producer/consumer ownership and enough helper throughput. Earlier full precomputation was slower, so avoid repeating a global materialize-and-barrier design. Benefits may vanish at H24/H48; verify those shapes too. |
| 4 | **Parallelize the chunk-state recurrence with affine composition.** Express each chunk as S_out = M_chunk S_in + B_chunk, then scan chunk transforms. | Removes the long per-head serial chain and exposes sequence parallelism when head count is small. This is the broadest algorithmic route to a larger gain. | More 128×128 products, scratch and communication; long products introduce their own rounding concerns. Prototype only with explicit work/traffic accounting and long-context FP64 checks. No speedup is established. |

Simple head/value-column sharding is not automatically a win: splitting 16
heads into two workers each creates 32 workers on 24 groups, hence two waves,
while duplicating some solve work and shrinking GEMMs. It needs an actual
balanced schedule and reuse plan. Likewise, using cubes for the 16×16 leaf solve
only helps if stable arithmetic and operand-transfer cost beat the current
vector substitution; returning to huge cancelling inverse powers is not a safe
substitute.

The setup clock is only about {100*phase1/(phase1+phase2):.1f}% of the diagnostic's
setup-plus-persistent time. Even deleting all setup has an optimistic Amdahl
ceiling of roughly {(phase1+phase2)/phase2:.2f}×. Deleting the measured feedback
wait alone would have an optimistic ceiling of roughly
{(phase1+phase2)/(phase1+phase2-cube[13]):.2f}×. These are illustrative limits under
fixed other costs, **not predictions**. A genuine 2× requires reducing multiple
dependencies or changing the recurrence/solve structure.

## Reproduce and inspect the evidence

```bash
# Fresh hardware profiles, production simulator trace, isolated diagnostic,
# then figures and Markdown. NPU runs are serialized.
bash scripts/profile_gdn_mega.sh collect

# Regenerate figures/report from this run's saved raw profiles:
bash scripts/profile_gdn_mega.sh render {report['raw_directory']}
```

The script uses the local `../traceviz-mcp` library (commit
`{report['traceviz_commit']}`) for parsing, overlap analysis and timeline rendering.
Its loader converts Chrome trace timestamps from microseconds to nanoseconds;
the JSON `displayTimeUnit` hint does not change the underlying timestamp unit.

- [Raw hardware tables, all clock samples, numerical checks and fingerprints](data/profile.json)
- [Traceviz summary and overlap calculations](data/profile_traceviz.txt)
- Vector figures: [pipeline](figures/profile_pipeline.svg), [hardware counters](figures/profile_hardware.svg), [groups](figures/profile_groups.svg), [stage scopes](figures/profile_stages.svg)
- [Reproduction script](../../scripts/profile_gdn_mega.sh)
- [Diagnostic generation, measurement, figures and report code](../../benchmarking/profile_gdn_mega.py)

Large raw profiler dumps remain in `{report['raw_directory']}` (ignored), including
the simulator trace and measurement inputs needed to render the report.
Diagnostic sources/libraries are disposable and can be regenerated by the
collection script. Committed evidence retains all hardware CSV contents, diagnostic samples and hashes; the trace hash
identifies the local raw input used for the figures. Recollection requires the
NPU/CANN environment and sibling traceviz checkout. Rendering requires the saved
raw directory and plotting dependencies. No diagnostic code or macros were
added to the production kernel.
'''
    path.write_text(text)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare-diagnostic', 'measure-diagnostic', 'render'])
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare-diagnostic': prepare_diagnostic(args.path)
    elif args.mode == 'measure-diagnostic': measure_diagnostic(args.path)
    else: render_report(args.path)


if __name__ == '__main__':
    main()
