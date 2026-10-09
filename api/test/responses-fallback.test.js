/* Offline regressions for the compatible adapter's bounded Responses fallback. */
import test from 'node:test';
import assert from 'node:assert/strict';
import compatible from '../coach/core/adapters/compatible.js';
import openai from '../coach/core/adapters/openai.js';
import { responsesSpec } from '../coach/core/adapters/responses.js';
import { SYSTEM_PROMPT } from '../coach/core/system-prompt.js';
import { attemptOnce } from '../coach/core/pipeline.js';

const ANSWER = '{"coach_contract":1,"nochange":true,"reading":"fine"}';
const cfgFor = baseUrl => ({ provider: 'compatible', providerOptions: { compatible: {
  baseUrl, headers: { 'x-routing-test': 'offline' }
} } });
const cfg = cfgFor('https://example.invalid');
const error = (message, status = 400) => ({ status, body: { error: { message } } });
const protocol = error('Model does not support this protocol');
const message = content => ({ type: 'message', content });
const output = text => ({ type: 'output_text', text });
const completed = { status: 200, body: { status: 'completed', output: [message([output(ANSWER)])] } };
function fakeFetch(answers) {
  const calls = [];
  const fetch = async (url, init) => {
    if (init.signal.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
    calls.push({ url, headers: init.headers, body: JSON.parse(init.body) });
    const answer = typeof answers === 'function' ? answers(calls.length, init) : answers[calls.length - 1];
    assert.ok(answer, 'no request beyond the scripted retry limit');
    return { ok: answer.status >= 200 && answer.status < 300, status: answer.status,
      text: async () => JSON.stringify(answer.body) };
  };
  fetch.calls = calls;
  return fetch;
}
const invoke = (fetch, overrides = {}) => compatible.invoke({ cfg, prompt: 'P', model: 'offline-model',
  env: { OPENAI_COMPAT_API_KEY: 'TEST_KEY' }, fetch, retryDelayMs: 0, ...overrides });

test('Responses fallback preserves versioned URLs, budget, headers and schema', async () => {
  for (const base of ['https://example.invalid', 'https://example.invalid/api/v1', 'https://example.invalid/paas/v4']) {
    const fetch = fakeFetch([protocol, completed]);
    const schema = { type: 'object', properties: { reading: { type: 'string' } } };
    const result = await invoke(fetch, { cfg: cfgFor(base), maxTokens: 48000, system: 'CONTEXT', schema });
    assert.equal(result.code, 0);
    assert.equal(result.text, ANSWER);
    const prefix = base === 'https://example.invalid' ? base + '/v1' : base;
    assert.deepEqual(fetch.calls.map(c => c.url), [prefix + '/chat/completions', prefix + '/responses']);
    assert.equal(fetch.calls[0].body.max_tokens, 48000);
    assert.equal(fetch.calls[1].body.max_output_tokens, 48000);
    assert.deepEqual(fetch.calls[1].body.reasoning, { effort: 'low' });
    assert.deepEqual(fetch.calls[1].body.text.format, { type: 'json_schema', name: 'coach_answer', schema });
    assert.deepEqual(fetch.calls[1].body.input, [
      { role: 'system', content: SYSTEM_PROMPT + '\n\nCONTEXT' }, { role: 'user', content: 'P' }
    ]);
    for (const call of fetch.calls) {
      assert.equal(call.headers['x-routing-test'], 'offline');
      assert.equal(call.headers.authorization, 'Bearer TEST_KEY');
      assert.equal(call.headers['content-type'], 'application/json');
    }
  }
});

test('each protocol gets one JSON-format retry, preserving the same configured budget', async () => {
  const fetch = fakeFetch([error('response_format unsupported'), protocol, error('json_schema unsupported'), completed]);
  const result = await invoke(fetch, { maxTokens: 2345 });
  assert.equal(result.code, 0);
  assert.equal(fetch.calls.length, 4);
  assert.equal(fetch.calls[0].body.response_format.type, 'json_object');
  assert.equal(fetch.calls[1].body.response_format, undefined);
  assert.equal(fetch.calls[2].body.text.format.type, 'json_object');
  assert.equal(fetch.calls[3].body.text.format.type, 'text');
  assert.deepEqual(fetch.calls.map(c => c.body.max_tokens ?? c.body.max_output_tokens), [2345, 2345, 2345, 2345]);
});

test('unsupported reasoning is retried once without it and preserves JSON fallback', async () => {
  for (const rejection of ['unsupported reasoning', "Unsupported parameter: 'reasoning'.", 'reasoning is not supported', 'This model does not support reasoning']) {
    const fetch = fakeFetch([protocol, error(rejection), error('json_schema unsupported'), completed]);
    const result = await invoke(fetch);
    assert.equal(result.code, 0, rejection);
    assert.equal(fetch.calls.length, 4);
    assert.deepEqual(fetch.calls[1].body.reasoning, { effort: 'low' });
    assert.equal(fetch.calls[2].body.reasoning, undefined);
    assert.equal(fetch.calls[2].body.text.format.type, 'json_object');
    assert.equal(fetch.calls[3].body.reasoning, undefined);
    assert.equal(fetch.calls[3].body.text.format.type, 'text');
  }
});

test('reasoning removal preserves a JSON-format retry already consumed on Responses', async () => {
  const fetch = fakeFetch([protocol, error('json_schema unsupported'), error('unsupported reasoning'), completed]);
  assert.equal((await invoke(fetch)).code, 0);
  assert.equal(fetch.calls[3].body.text.format.type, 'text');
  assert.equal(fetch.calls[3].body.reasoning, undefined);
});

test('ordinary reasoning errors and non-400 unsupported errors never strip options', async () => {
  for (const [text, status] of [['invalid reasoning budget', 400], ['reasoning failed', 400], ['unsupported reasoning', 422], ['unsupported reasoning', 401]]) {
    const fetch = fakeFetch([protocol, error(text, status)]);
    const result = await invoke(fetch);
    assert.equal(result.code, 1);
    assert.equal(fetch.calls.length, 2);
    assert.equal(result.stderr, `${status} ${text}`);
  }
});

test('repeated protocol, reasoning and JSON-format errors all terminate', async () => {
  for (const responses of [
    [protocol, protocol],
    [protocol, error('unsupported reasoning'), error('unsupported reasoning')],
    [protocol, error('json_schema unsupported'), error('json_schema unsupported')]
  ]) {
    const fetch = fakeFetch(responses);
    assert.equal((await invoke(fetch)).code, 1);
    assert.equal(fetch.calls.length, responses.length);
  }
});

test('protocol switching is compatible-only and only on its exact 400 error', async () => {
  for (const first of [error('bad request'), error('Model does not support this protocol', 422)]) {
    const fetch = fakeFetch([first]);
    assert.equal((await invoke(fetch)).code, 1);
    assert.equal(fetch.calls.length, 1);
  }
  const fetch = fakeFetch([protocol]);
  const result = await openai.invoke({ cfg: {}, model: 'offline-model', prompt: 'P', env: { OPENAI_API_KEY: 'TEST_KEY' }, fetch });
  assert.equal(result.code, 1);
  assert.equal(fetch.calls.length, 1);
});

test('transient retries share one budget across both protocols', async () => {
  const fetch = fakeFetch([
    error('busy', 429), error('response_format unsupported'), protocol,
    error('busy', 503), error('json_schema unsupported'),
    error('unsupported reasoning'), error('busy', 503)
  ]);
  const result = await invoke(fetch);
  assert.equal(result.code, 1);
  assert.equal(result.stderr, '503 busy');
  assert.equal(fetch.calls.length, 7);
});

test('Responses transient retry can recover without resetting protocol or format', async () => {
  const fetch = fakeFetch([protocol, error('busy', 503), completed]);
  assert.equal((await invoke(fetch)).code, 0);
  assert.equal(fetch.calls.length, 3);
  assert.deepEqual(fetch.calls[1], fetch.calls[2]);
});

test('already-aborted invokes send no request', async () => {
  const control = new AbortController();
  control.abort();
  const fetch = fakeFetch([]);
  const result = await invoke(fetch, { signal: control.signal });
  assert.equal(result.timedOut, true);
  assert.equal(fetch.calls.length, 0);
});

test('abort during transient retry delay prevents the next request', async () => {
  const control = new AbortController();
  const fetch = fakeFetch(() => {
    setTimeout(() => control.abort(), 0);
    return error('busy', 503);
  });
  const result = await invoke(fetch, { signal: control.signal, retryDelayMs: 100 });
  assert.equal(result.timedOut, true);
  assert.equal(fetch.calls.length, 1);
});

test('abort before protocol, reasoning or JSON fallback sends nothing further', async () => {
  for (const final of [protocol, error('unsupported reasoning'), error('json_schema unsupported')]) {
    const control = new AbortController();
    const responses = final === protocol ? [final] : [protocol, final];
    const fetch = fakeFetch(n => {
      if (n === responses.length) control.abort();
      return responses[n - 1];
    });
    const result = await invoke(fetch, { signal: control.signal });
    assert.equal(result.timedOut, true);
    assert.equal(fetch.calls.length, responses.length);
  }
});

test('refusal wins over convenience output_text and ordinary output fragments', async () => {
  for (const output_text of [undefined, ANSWER]) {
    const fetch = fakeFetch([protocol, { status: 200, body: { status: 'completed', output_text,
      output: [message([output(ANSWER), { type: 'refusal', refusal: 'declined' }])] } }]);
    const result = await invoke(fetch);
    assert.equal(result.code, 1);
    assert.equal(result.text, '');
    assert.match(result.stderr, /refused: declined/);
  }
});

test('any incomplete answer fails, including content-filtered or reasoning-only answers', async () => {
  for (const reason of [undefined, 'max_output_tokens', 'content_filter', 'other_reason']) {
    for (const wire of [{ output: [{ type: 'reasoning' }] }, { output: [message([output(ANSWER)])], output_text: ANSWER }]) {
      const fetch = fakeFetch([protocol, { status: 200, body: { status: 'incomplete', incomplete_details: { reason }, ...wire } }]);
      const result = await invoke(fetch);
      assert.equal(result.code, 1);
      assert.equal(result.text, '');
      assert.match(result.stderr, reason && reason !== 'max_output_tokens' ? /incomplete/ : /cut off/);
    }
  }
});

test('completed Responses text supports fragments and gateway convenience output', () => {
  const spec = responsesSpec('compatible');
  assert.equal(spec.readText({ status: 'completed', output: [message([output('{"a":'), output('1}')])] }).text, '{"a":1}');
  assert.equal(spec.readText({ status: 'completed', output_text: ANSWER }).text, ANSWER);
  assert.match(spec.readText(null).error, /empty/);
});

const responseShapes = [
  ['output messages', { output: [message([output(ANSWER)])] }],
  ['convenience output_text', { output_text: ANSWER }]
];
const pipelineAttempt = fetch => attemptOnce({
  adapter: compatible, cfg, kind: 'review', model: 'offline-model',
  payload: { plan: { routines: [], week: {} } },
  invokeOpts: { env: { OPENAI_COMPAT_API_KEY: 'TEST_KEY' }, fetch, retryDelayMs: 0 }
}, null);

for (const status of ['cancelled', 'in_progress', 'queued']) {
  for (const [shape, wire] of responseShapes) {
    test(`Responses ${status} with ${shape} is a provider failure, even with valid review JSON`, async () => {
      const fetch = fakeFetch([protocol, { status: 200, body: { status, ...wire } }]);
      const result = await invoke(fetch);
      assert.equal(result.code, 1);
      assert.equal(result.text, '');
      assert.match(result.stderr, /not completed/);
      assert.equal(fetch.calls.length, 2);
    });

    test(`pipeline never approves a nochange review from ${status} with ${shape}`, async () => {
      const fetch = fakeFetch([protocol, { status: 200, body: { status, ...wire } }]);
      const result = await pipelineAttempt(fetch);
      assert.equal(result.ok, false);
      assert.equal(result.errorClass, 'provider');
      assert.equal(result.nochange, undefined);
      assert.equal(result.repairable, undefined);
      assert.match(result.detail, /not completed/);
      assert.equal(fetch.calls.length, 2);
    });
  }
}

for (const status of ['completed', undefined]) {
  for (const [shape, wire] of responseShapes) {
    test(`Responses ${status ?? 'without status'} with ${shape} remains a valid nochange review`, async () => {
      const body = status === undefined ? wire : { status, ...wire };
      const fetch = fakeFetch([protocol, { status: 200, body }]);
      const result = await pipelineAttempt(fetch);
      assert.equal(result.ok, true);
      assert.equal(result.nochange, true);
      assert.equal(result.reading, 'fine');
      assert.equal(fetch.calls.length, 2);
    });
  }
}
