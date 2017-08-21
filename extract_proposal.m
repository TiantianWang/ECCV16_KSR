function [masks]=extract_proposal(img)
%{
    Copyright (c) 2014, Philipp Krähenbühl
    All rights reserved.
	
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.
	
    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%}
% clear all;
init_gop;

% Set a boundary detector by calling (before creating an OverSegmentation!):
% gop_mex( 'setDetector', 'SketchTokens("../data/st_full_c.dat")' );
% gop_mex( 'setDetector', 'StructuredForest("../data/sf.dat")' );
gop_mex( 'setDetector', 'MultiScaleStructuredForest("./gop_1.3/matlab/data/sf.dat")' );

% Setup the proposal pipeline (baseline)
p = Proposal('max_iou', 0.8,...
             'unary', 130, 5, 'seedUnary()', 'backgroundUnary({0,15})',...
             'unary', 130, 1, 'seedUnary()', 'backgroundUnary({})', 0, 0, ... % Seed Proposals (v1.2 and newer)
             'unary', 0, 5, 'zeroUnary()', 'backgroundUnary({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})' ... % Background Proposals
             );

os = OverSegmentation(img);
% Generate proposals
props = p.propose( os );
[rows,columns,dim]=size(img);
num_imgpixels=rows*columns;
j=1;
for i=1:size(props,1)
    
    mask = props(i,:);
    I_pro=double(mask( os.s()+1 ));
    num_propixels=sum(I_pro(:));
    sum_pro=[sum(I_pro(:,1)) sum(I_pro(:,columns)) sum(I_pro(1,:)) sum(I_pro(rows,:))];
    num_zero=sum(sum_pro==0);  
    if (num_zero==1 ||num_zero==2 || num_zero==3||num_zero==4) && (num_propixels>num_imgpixels*0.02) && (num_propixels<num_imgpixels*0.7)
        masks(:,:,j)=I_pro;
        j=j+1;
    end
end
return;