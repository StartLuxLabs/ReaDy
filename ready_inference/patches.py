"""Verification attention and natural-stop handling used by this inference configuration."""


def install():
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.managers.schedule_batch import Req, FINISH_LENGTH, FINISH_MATCHED_TOKEN
    forward = TritonAttnBackend.forward_extend
    finish = Req.update_finish_state

    def verify(self, *args, **kwargs):
        batch = kwargs.get('forward_batch')
        if batch is None:
            batch = args[4]
        if not self.is_draft_runner and batch.forward_mode.is_target_verify():
            previous = self.enable_deterministic
            self.enable_deterministic = True
            try:
                return forward(self, *args, **kwargs)
            finally:
                self.enable_deterministic = previous
        return forward(self, *args, **kwargs)

    def update(self, new_accepted_len=1):
        previous = self.finished() or bool(self.to_finish)
        result = finish(self, new_accepted_len)
        if previous or not isinstance(self.finished_reason, FINISH_LENGTH) or self.sampling_params.ignore_eos:
            return result
        if self.sampling_params.stop_strs or self.sampling_params.stop_regex_strs:
            return result
        stop = set(self.sampling_params.stop_token_ids or ()) | set(self.eos_token_ids or ())
        if self.tokenizer is not None:
            stop.add(self.tokenizer.eos_token_id)
            stop.update(self.tokenizer.additional_stop_token_ids or ())
        begin = max(0, len(self.output_ids) - new_accepted_len)
        end = min(len(self.output_ids), self.sampling_params.max_new_tokens)
        for index in range(begin, end):
            if self.output_ids[index] in stop:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[index])
                self.finished_len = index + 1
                break
        return result

    TritonAttnBackend.forward_extend = verify
    Req.update_finish_state = update
