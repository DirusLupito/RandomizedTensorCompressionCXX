function measureFlow(dataset, dataDir, outputDir)
% Run the serial baseline and, when already open, the current parallel pool.
% TeBaC and its dependencies must be on the user's MATLAB path.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(dataDir), dataDir = fullfile(root, 'data'); end
if isempty(outputDir), outputDir = fullfile(root, 'build', 'measuredExperiments'); end
if ~isfolder(outputDir), mkdir(outputDir); end

% The two planar flows use TeBaC's height x time x width convention.
% Tangaroa uses every x-y point, time step and depth plane of u.
path = fullfile(dataDir, [dataset '.nc']);
if strcmp(dataset, 'tangaroa')
    info = ncinfo(path, 'u');
    assert(isequal({info.Dimensions.Name}, {'xdim','ydim','zdim','tdim'}), ...
        'Expected Tangaroa u in MATLAB x, y, z, t order.');
    nx = info.Size(1); ny = info.Size(2); nz = info.Size(3); nt = info.Size(4);
    T = zeros(nx*ny, nt, nz);
    for t = 1:nt
        frame = ncread(path, 'u', [1 1 1 t], [nx ny nz 1]);
        T(:,t,:) = reshape(frame, nx*ny, 1, nz);
    end
else
    T = extractNetCDFAsTensor(path);
end
M = generateUnitary(size(T,3), 'dct');

% gcp('nocreate') leaves pool setup to the caller. Even a one-worker pool
% gets a distinct CSV from the ordinary for-loop serial implementation.
pool = gcp('nocreate');
runSweep(dataset, T, M, outputDir, 'serial', 1);
if ~isempty(pool)
    runSweep(dataset, T, M, outputDir, 'parallel', pool.NumWorkers);
end
end

function runSweep(dataset, T, M, outputDir, execution, threads)
% Record each of five independent calls. Closing the partial file before
% renaming it keeps the Python collector from plotting an interrupted sweep.
[m,p,n] = size(T);
csv = fullfile(outputDir, sprintf('%s_matlab_%s_%d.csv', dataset, execution, threads));
[~,runId] = fileparts(tempname(outputDir));
partial = [csv '.' runId '.partial'];
[fid, message] = fopen(partial, 'w');
if fid < 0, error('Cannot open %s: %s', partial, message); end
closer = onCleanup(@() fclose(fid));
fprintf(fid, 'dataset,language,sweep,algorithm,execution,threads,batch_size,c,gamma,repeat,seconds,m,p,n\n');
fprintf('%s, %s (%d): %s\n', dataset, execution, threads, partial);
if strcmp(execution,'serial')
    decompose = @TSVDMserial;
else
    decompose = @TSVDM;
end
for c = 1:min(m,p)
    rho = repmat(c,n,1);
    for rep = 1:5
        start = tic;
        [U,S,V] = decompose(T,M,true,'econ');
        [Ut,St,Vt] = truncateEconomy(U,S,V,rho);
        seconds = toc(start);
        fprintf(fid,'%s,matlab,scaling,TSVDM,%s,%d,,%d,,%d,%.17g,%d,%d,%d\n', ...
            dataset,execution,threads,c,rep,seconds,m,p,n);
        clear U S V Ut St Vt
    end
end
for gamma = [0.9 0.99 0.999 0.9999]
    for rep = 1:5
        start = tic;
        if strcmp(execution,'serial')
            [U,S,V,rho] = serialTsvdm2(T,M,gamma);
        else
            [U,S,V,rho] = TSVDM2(T,M,gamma,true,'econ');
        end
        seconds = toc(start);
        fprintf(fid,'%s,matlab,scaling,TSVDM2,%s,%d,,,%.4g,%d,%.17g,%d,%d,%d\n', ...
            dataset,execution,threads,gamma,rep,seconds,m,p,n);
        clear U S V rho
    end
end
clear closer
movefile(partial,csv,'f');
fprintf('Saved %s\n',csv);
end

function [Ut,St,Vt] = truncateEconomy(U,S,V,rho)
% As in truncateTSVDM, copy the first rho(i) triples of each slice. For an
% economy SVD of a rectangular matrix, U and V supply their own row counts.
r = max(rho);
n = numel(rho);
Ut = zeros(size(U,1),r,n);
St = zeros(r,r,n);
Vt = zeros(size(V,1),r,n);
for i = 1:n
    k = 1:rho(i);
    Ut(:,k,i) = U(:,k,i);
    St(k,k,i) = S(k,k,i);
    Vt(:,k,i) = V(:,k,i);
end
end

function [U,S,V,rho] = serialTsvdm2(T,M,gamma)
% Follow TSVDM2's global cutoff and keep ties across slices. Using TSVDMserial
% here keeps the baseline serial even when the caller has a parallel pool.
[U,S,V] = TSVDMserial(T,M,true,'econ');
n = size(S,3);
delta = zeros(size(S,1),n);
for i = 1:n
    delta(:,i) = diag(S(:,:,i)).^2;
end
delta = sort(delta(:),'descend');
w = cumsum(delta);
cutoff = inf;
if w(end) > 0
    j = find(w/w(end) > gamma, 1);
    cutoff = sqrt(delta(j));
end
rho = zeros(n,1);
for i = 1:n
    rho(i) = sum(diag(S(:,:,i)) >= cutoff);
    S(rho(i)+1:end,rho(i)+1:end,i) = 0;
end
end
