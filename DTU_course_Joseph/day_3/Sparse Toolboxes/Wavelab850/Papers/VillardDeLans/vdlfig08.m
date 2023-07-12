% vdlfig08 -- Show MP phase plane for Tweet signal
%
% The phase plane generated by the Matching Pursuits algorithm
% on the tweet signal is markedly less clear than that found by
% BOB..  In this case, MP is too adaptive.
%

	tweet = ReadSignal('Tweet');
	n = length(tweet);
%
	[atomic,resid] = CPPursuit(tweet,6,'Sine',100,.001,0);
%
	ImageAtomicPhase('CP',atomic,n,[]);
	axis([ 0 1 .4 .6]);
	title('Figure 8: MP Phase Plane ; Tweet');

% 
% Copyright (c) 1995, Jonathan Buckheit.
% Prepared for ``WaveLab and Reproducible Research''
% for XV Recontres Franco-Belges symposium proceedings.
%
    
    
 
 
%
%  Part of Wavelab Version 850
%  Built Tue Jan  3 13:20:42 EST 2006
%  This is Copyrighted Material
%  For Copying permissions see COPYING.m
%  Comments? e-mail wavelab@stat.stanford.edu 