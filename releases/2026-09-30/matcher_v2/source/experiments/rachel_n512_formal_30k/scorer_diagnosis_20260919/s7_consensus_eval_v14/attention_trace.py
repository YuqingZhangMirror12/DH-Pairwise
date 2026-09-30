"""Read-only hooks for one future frozen C10 forward, never a training patch.

Reconstruct softmax from the ACTUALLY EXECUTED Q/K/V projections, measure and
bias. Verify weights@V against the actual input to the output projection.
Only head means are retained. They are attention, not Matcher Q, calibrated
correctness, gradients, or a causal attribution. Only the terminal-gated C10
case path uses this helper; it cannot launch inference by itself.
"""
import inspect
import math

import torch


class AttentionTrace:
    """Capture detached per-call head means without changing any forward return.

Records retain packed batch/query/key axes and masks. Call ordinals are NOT
silently relabeled as cluster, A/B, initial/final, or original contour IDs;
the export adapter must explicitly validate and bind those identities later.
"""
    def __init__(self, head):
        self.head = head
        self.records = []
        self.handles = []
        self.active = {}
        self.counts = {}
        self.used = False

    def __enter__(self):
        if self.used or self.handles:
            raise ValueError('attention capture is single-use')
        if self.head.training or torch.is_grad_enabled():
            raise ValueError('attention trace requires frozen eval/no_grad inference')
        if any(m._forward_hooks or m._forward_pre_hooks for m in self.head.modules()):
            raise ValueError('refuse ambiguous pre-existing forward hooks')
        modules = [(name, m) for name,m in self.head.named_modules()
                   if type(m).__name__ == 'MeasureAttention']
        if not modules:
            raise ValueError('no registered MeasureAttention modules')
        self.used = True
        try:
            for name, module in modules:
                self.handles.append(module.register_forward_pre_hook(
                    self._start(name), with_kwargs=True))
                for key in ('q','k','v'):
                    self.handles.append(getattr(module,key).register_forward_hook(
                        self._projection(name,key)))
                self.handles.append(module.out.register_forward_pre_hook(self._pre_output(name)))
                self.handles.append(module.register_forward_hook(self._finish(name)))
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active.clear()

    def __exit__(self, *exc):
        self.close()
        return False

    def _start(self, name):
        def hook(module, args, kwargs):
            if name in self.active:
                raise ValueError('reentrant attention call is not supported')
            values = inspect.signature(module.forward).bind(*args, **kwargs)
            values.apply_defaults()
            self.active[name] = {key:(value.detach() if isinstance(value,torch.Tensor) else value)
                                 for key,value in values.arguments.items()}
        return hook

    def _projection(self, name, key):
        def hook(module, args, output):
            # Do not call the projection twice or change its return value.
            if key+'_executed' in self.active[name]:
                raise ValueError('projection unexpectedly ran twice')
            self.active[name][key+'_executed'] = output.detach()
        return hook

    def _pre_output(self, name):
        def hook(module, args):
            self.active[name]['actual_pre_output'] = args[0].detach()
        return hook

    def _finish(self, name):
        def hook(module, args, output):
            values = self.active.pop(name)
            single = values['query'].ndim == 2
            def batched(key):
                v = values[key]
                return v[None] if single and v is not None else v
            query, key = batched('query'), batched('key')
            qv, kv, measure = (batched(k) for k in ('query_valid','key_valid','key_measure'))
            bias = batched('geometry_bias')
            batch,n,dim = query.shape
            m = key.shape[1]
            if query.dtype != torch.float32 or key.dtype != torch.float32:
                raise ValueError('registered trace supports the current FP32 protocol only')
            if not torch.isfinite(measure).all() or (measure < 0).any():
                raise ValueError('invalid key observation measure')
            if not n or not m:
                if any(k.endswith('_executed') for k in values):
                    raise ValueError('empty attention unexpectedly projected inputs')
                mean = query.new_zeros((batch,n,m))
                error = 0.
            else:
                q = values['q_executed'].reshape(batch,n,module.heads,module.width).transpose(1,2)
                k = values['k_executed'].reshape(batch,m,module.heads,module.width).transpose(1,2)
                v = values['v_executed'].reshape(batch,m,module.heads,module.width).transpose(1,2)
                logits = (q@k.transpose(-1,-2))/math.sqrt(module.width)
                logits = logits+measure.clamp_min(1e-30).log()[:,None,None]
                if bias is not None:
                    logits = logits+bias[:,None]
                logits = logits.masked_fill(~kv[:,None,None], -torch.inf)
                any_keys = kv.any(-1)
                logits = torch.where(any_keys[:,None,None,None],logits,torch.zeros_like(logits))
                weights = logits.float().softmax(-1).to(v.dtype)
                pre_output = (weights@v).transpose(1,2).reshape(batch,n,dim)
                actual = values['actual_pre_output']
                if not torch.isfinite(weights).all() or not torch.allclose(
                        pre_output, actual, rtol=0., atol=2e-6):
                    raise ValueError('attention reconstruction differs from executed weights@V')
                error = float((pre_output-actual).abs().max())
                mean = weights.mean(1)
            effective = torch.where((qv & kv.any(-1)[:,None])[...,None],mean,torch.zeros_like(mean))
            ordinal = self.counts.get(name,0)
            self.counts[name] = ordinal+1
            def cpu(value):
                return value.detach().cpu().clone()
            self.records.append(dict(module=name, call_ordinal=ordinal, heads=module.heads,
                query_valid=cpu(qv), key_valid=cpu(kv), key_measure=cpu(measure),
                geometry_bias=None if bias is None else cpu(bias),
                softmax_head_mean=cpu(mean), effective_head_mean=cpu(effective),
                reconstruction_max_abs=error, input_was_single=single,
                meaning='reconstructed from executed projected Q/K/V; head mean, not Matcher Q',
                invalid_query_or_all_invalid_keys='effective mean is zero; raw internal softmax may be uniform'))
        return hook


def attach_trace(meta, store, prediction, trace):
    """Bind verified packed axes to exact cluster/stage/side/contour identities."""
    if not trace.used or trace.handles or trace.active:
        raise ValueError('attention trace must finish and remove hooks before serialization')
    count = len(prediction.clusters)
    if len(trace.records) != (16 if count else 0):
        raise ValueError('expected exactly two 2-layer bidirectional scoring passes')
    meta['attention_capture'] = dict(schema='s7-consensus-executed-attention/1',
        method='reconstruct softmax from executed projections; verify against executed pre-output weights@V',
        heads=4, aggregation='arithmetic mean of four heads, not all per-head matrices',
        attention_calls=len(trace.records), no_additional_network_forward=True,
        geometry_bias_meaning=meta['semantics'].get('kernels'),
        interpretation='feature-reading weights, NOT Matcher Q, calibrated correctness or causal attribution')
    meta['semantics']['attention_weights_exported'] = bool(count)
    used = set()
    for record in trace.records:
        parts = record['module'].split('.')
        if (len(parts)!=3 or parts[0]!='layers' or parts[1] not in ('0','1')
                or parts[2] not in ('self_attention','cross_attention') or record['heads']!=4):
            raise ValueError('unregistered attention architecture')
        ordinal=record['call_ordinal']
        if ordinal not in range(4):
            raise ValueError('unregistered attention call order')
        stage='initial' if ordinal<2 else 'final'
        side='a' if ordinal%2==0 else 'b'
        key_side=side if parts[2]=='self_attention' else ('b' if side=='a' else 'a')
        label=parts[1]+'/'+parts[2]+'/'+side
        identity=(stage,label)
        if identity in used:
            raise ValueError('duplicate attention stage/operator')
        used.add(identity)
        evidences=[(c.initial_encoded if stage=='initial' else c.encoded).evidence
                   for c in prediction.clusters]
        def indices(which):
            return [getattr(e,which).valid.nonzero(as_tuple=False).flatten().detach().cpu()
                    for e in evidences]
        qi,ki=indices(side),indices(key_side)
        qw,kw=max(1,max(map(len,qi))),max(1,max(map(len,ki)))
        expected_q=torch.arange(qw)[None]<torch.tensor(list(map(len,qi)))[:,None]
        expected_k=torch.arange(kw)[None]<torch.tensor(list(map(len,ki)))[:,None]
        if (not torch.equal(record['query_valid'],expected_q)
                or not torch.equal(record['key_valid'],expected_k)
                or record['effective_head_mean'].shape!=(count,qw,kw)):
            raise ValueError('captured attention packing differs from actual evidence')
        for i,(saved,evidence) in enumerate(zip(meta['clusters'],evidences)):
            nq,nk=len(qi[i]),len(ki[i])
            key_evidence=getattr(evidence,key_side)
            expected_measure=(key_evidence.mass*key_evidence.observed_arc_px).detach().cpu()[ki[i]]
            expected_measure=torch.nn.functional.pad(expected_measure,(0,kw-nk))
            if not torch.equal(record['key_measure'][i],expected_measure):
                raise ValueError('captured attention key measure differs from actual evidence')
            bias=record['geometry_bias']
            if parts[2]=='self_attention':
                if bias is not None:
                    raise ValueError('self attention unexpectedly has geometric bias')
            else:
                kernels=evidence.kernels.detach().cpu()
                if side=='b':kernels=kernels.T
                expected_bias=kernels[qi[i]][:,ki[i]].clamp_min(1e-30).log()
                expected_bias=torch.nn.functional.pad(expected_bias,(0,kw-nk,0,qw-nq))
                if bias is None or not torch.allclose(bias[i],expected_bias,rtol=2e-6,atol=2e-6):
                    raise ValueError('attention bias differs from this cluster and stage')
            base=f'cluster_{saved["cluster_id"]:03d}/{stage}/attention/{label}'
            exported=dict(module=record['module'],layer=int(parts[1]),query_side=side,key_side=key_side,
                kind=parts[2],heads=4,call_ordinal=ordinal,
                reconstruction_max_abs=record['reconstruction_max_abs'],
                query_compact_ids=store.add(base+'/query_compact_ids',qi[i]),
                key_compact_ids=store.add(base+'/key_compact_ids',ki[i]),
                key_measure=store.add(base+'/key_measure',record['key_measure'][i,:nk]),
                head_mean=store.add(base+'/head_mean',record['effective_head_mean'][i,:nq,:nk]),
                geometry_bias=None if bias is None else store.add(base+'/geometry_bias',bias[i,:nq,:nk]))
            saved['stages'][stage].setdefault('attention',{})[label]=exported
    if count and len(used)!=16:
        raise ValueError('incomplete attention operator capture')
