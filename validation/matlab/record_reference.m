function record_reference(matlab_repo, method, example, Ndim, seed, outdir)
% Run the ORIGINAL MATLAB code once and record every random number drawn.
%
%   record_reference('/path/to/Sequential-multiple-importance-sampling-for-Reliability-Estimation', ...
%                    'high', 'Ex10', 50, 1, '../reference')
%
% method  : 'low' (SeMIS_low_dimension, needs fitrgp, MATLAB only) |
%           'high' (SeMIS_high_dimension) | 'sus' (SuS baseline)
% example : 'Ex1' ... 'Ex12'   (Ndim is used by Ex10-Ex12 only)
% The functions randn/rand/randperm/randi/randsample in this folder shadow the
% built-in ones and log every draw; rec_*.mat holds the stream and res_*.mat
% the results. validate.py replays the stream through the Python port.
here = fileparts(mfilename('fullpath'));
addpath(matlab_repo);
addpath(fullfile(matlab_repo, 'Limit-state-functions'));
addpath(fullfile(matlab_repo, 'Support-Fun'));
addpath(fullfile(matlab_repo, 'Support-Fun', 'Operations on Logarithmic values'));
addpath(here);                         % shims first on the path
if exist('OCTAVE_VERSION', 'builtin')
    addpath(fullfile(here, 'octave'));  % Octave compatibility shims (makedist, pagemtimes, ...)
    pkg load statistics
    builtin('rand', 'seed', seed); builtin('randn', 'seed', seed);
else
    rng(seed);
end
if any(strcmp(example, {'Ex10', 'Ex11', 'Ex12'}))
    Tfun = feval(example, Ndim);
else
    Tfun = feval(example);
end
switch method
    case 'low',  IntBay = SeMIS_low_dimension(Tfun);
    case 'high', IntBay = SeMIS_high_dimension(Tfun);
    case 'sus',  IntBay = SuS(Tfun);
end
rec_reset();
semcs = SeMIS(IntBay, aCS);
semcs.NumSam = 1000;
out = semcs.RunIte;
Ncal = sum(out.Ncal); Nite = out.Nite;
u_last = out.x{end};
if strcmp(method, 'sus')
    Pf = IntBay.EvlPf(out); cv_lw = nan; cv_up = nan;
    l = cell2mat(out.g);
else
    mis = MIS_Rel(out); Pf = mis.Pf_hat;
    [cv_lw, cv_up] = mis.estimateCov;
    l = cellfun(@(s) s.l, out.g);
end
tag = sprintf('%s_%s_%d_%d', method, example, Ndim, seed);
rec_save(fullfile(outdir, ['rec_' tag '.mat']));
save(fullfile(outdir, ['res_' tag '.mat']), '-v7', 'Pf', 'cv_lw', 'cv_up', 'Ncal', 'Nite', 'l', 'u_last');
end
