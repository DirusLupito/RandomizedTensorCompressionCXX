function measureBoussinesq(dataDir, outputDir)
% Time TSVDM and TSVDM2 on the full u component of boussinesq.nc.
% With no arguments, use this project's data and timing directories. Pass
% dataDir and outputDir explicitly when running from a TeBaC checkout.
if nargin < 1, dataDir = ''; end
if nargin < 2, outputDir = ''; end
measureFlow('boussinesq', dataDir, outputDir);
end
