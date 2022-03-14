clear all;
%% SUPERIMPOSED ZADOFF CHU SEQUENCE DETECTION USING DEEP NEURAL NETWORK

% ========================= SYSTEM MODEL ==================================

% Average number of users(M)     : 18 (for 40 symbol duration i.e 10.67 ms)
% Desired success probability(p) : 0.9
% Bandwidth          : 180 Khz 
% Subcarrier spacing : 3.75 Khz
% Symbol duration    : 0.267 ms
% Resource grid      : 48 sub-carriers x 40 symbols


% * Number of users in the system is poisson distributed with mean of M.

% * To get a success probability of 0.9 (i.e no collision in preamble picking)
% we need N = ceil(M^2/(2*ln(1/p))) = 1538 different preambles. 

% * Instead of giving 1538 different choices over 40 symbol duration, entire 
% duration is divided into 40 slots each of 1 symbol duration. 

% * This implies that we need ceil(1538/40) = 39 different preambles per slot
% i.e, we need atleast 39 different ZC sequences for each slot. 

% * Here since we can transmit a 48 length sequence in each slot, we choose
% nearest prime number (<48) i.e, 47 to be the length of the sequence.

% * This setting gives us 47-1 = 46 different possible ZC sequences,
% which is greater than required number of sequences per slot i.e, 39

% * With 46 preambles per slot, we will be having a total of 46*40 = 1840
% different preambles over 40 slots/symbol duration.

% * With these many number of preambles we will get success 91.5% of the time.
   
% Any user who wants to send its data will pick a slot uniformly from the
% available Num_slots = 40 slots. Next it will uniformly choose a preamble 
% from the available 46 preambles in that slot. With this system model,
% P(k users picking same slot)   = M^C_k * (1/Num_slots)^k * (1- 1/Num_slots)^(M-k) 
% P(0 user(s) picking same slot) = 0.634
% P(1 user(s) picking same slot) = 0.2926
% P(2 user(s) picking same slot) = 0.0637
% P(3 user(s) picking same slot) = 0.0087

% ====================== PROBLEM STATEMENT ================================

% The problem at the base station is to detect the presence of different 
% preambles from the received superimposed signal using neural network.
% Once the detection is over, next step is to estimate timing and frequency
% offset of the detected users.

% ============================= REFERENCE =================================

% Title : On Preamble-based Grant-Free Transmission in Low Power Wide Area 
% (LPWA) IoT Networks.
% Authors : Harini G S, Mysore Balasubramanya Naveen and Rana Mrinal

%% ZADOFF CHU SEQUENCE GENERATION

Num_SC           = 48; % Number of sub carriers
Tot_syms         = 40; % Total number of symbols
Num_sym_per_slot = 1;  % Number of symbols per slot
Num_slots        = Tot_syms/Num_sym_per_slot; 
L                = Num_SC*Num_sym_per_slot; % length of Zadoff chu sequence required

% check whether entered L is prime or not. 
% if not find the nearest prime number 
primL = L;
while ~isprime(primL)   
    primL = primL - 1;   
end

NUM_ZC_SEQ   = primL - 1; % Number of possible ZC sequences
NUM_SEQ_REQ  = 39;        % Number of ZC sequences required    

% Creating "NUM_ZC_SEQ" ZC sequences of length "primL" 
% "L - primL" zeros are padded to make it "L" length sequence
preambs = zeros(L,NUM_ZC_SEQ);
for root = 1:NUM_ZC_SEQ        
    preambs(1:end-1,root) = zadoffChuSeq(root,primL);
end


%% PROPERTIES OF ZC SEQUENCES

% Norm of the ZC sequences (constant)
close all;
abs_preambs = sum(abs(preambs(1:NUM_ZC_SEQ+1,:)));
figure();
plot(abs_preambs,'r-o','LineWidth',2,...
    'MarkerSize',8,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','y');
title(sprintf("Norm of the ZC sequences of length %d",NUM_ZC_SEQ+1));
xlabel("Root of the ZC sequence");
ylabel("Norm of the sequence");

% Auto correlation
root       = 1;
preamb     = preambs(1:NUM_ZC_SEQ+1,root);
temp       = xcorr(preamb);
auto_corr  = abs(temp(NUM_ZC_SEQ+1:end,1));
norm_corr  = norm(temp(NUM_ZC_SEQ+1:end,1));
figure();
plot(auto_corr,'g-o','LineWidth',2,...
    'MarkerSize',8,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','m');
title(sprintf("Autocorrelation of ZC sequence of length %d and root %d",NUM_ZC_SEQ+1, root));
xlabel("Circular shift");
ylabel("Correlation value");

% Cross correlation
root        = 1;
preamb      = preambs(1:NUM_ZC_SEQ+1,root);
abs_corr    = zeros(NUM_ZC_SEQ,1);
for i = 1:NUM_ZC_SEQ
    preamb_1         = preambs(1:NUM_ZC_SEQ+1,i);
    temp             = xcorr(preamb,preamb_1);
    abs_corr(i,1)    = abs(temp(NUM_ZC_SEQ+1,1));
end
figure();
plot(abs_corr,'b-o','LineWidth',2,...
    'MarkerSize',8,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','c');
title(sprintf("Crosscorrelation of root %d ZC sequence with ZC sequences of other roots i.e, [%d,%d] (length : %d)",root,1,NUM_ZC_SEQ,NUM_ZC_SEQ+1));
xlabel("Root of ZC sequence");
ylabel("Correlation value");

%% PARAMETERS FOR DATASET GENERATION 

% Other system parameters
SC_spacing   = 3750; % Sub carrier spacing (Hz)
N_fft        = 512;  % FFT/IFFT size
CP_len       = 32;  % Cyclic prefix length  
CP_len_test  = 32;   % Cyclic prefix length used in generation of test data
Num_antennas = 2;    % Number of receive antenna
N_rep      = 1;

% Choosing the channel condition/impairments for which data has to be generated
Channel_types = ["Single tap rayleigh", "EPA"];
Noise   = 1; % Additive white gaussian noise
Channel = 1; % Channel 
if Channel
   Channel_type = 2; 
end
if Channel_type == 2  
    % LTE fading channel model configuration
    chcfg.DelayProfile       = 'EPA';
    chcfg.NRxAnts            = Num_antennas;
    chcfg.DopplerFreq        = 1;
    chcfg.MIMOCorrelation    = 'Low';
    chcfg.Seed               = 1;
    chcfg.InitPhase          = 'Random';
    chcfg.ModelType          = 'GMEDS';
    chcfg.NTerms             = 16;
    chcfg.NormalizeTxAnts    = 'On';
    chcfg.NormalizePathGains = 'On';
    chcfg.SamplingRate       = SC_spacing*N_fft;
end
TOFO     = 1;   % Timing and frequency offset
FO_max  = 200; % Maximum frequency offset for training data

if Noise
    % SNR(in db) at which training data is generated 
    SNRdb_train   = 6;              
    SNR_train     = 10^(SNRdb_train/10);
    stdv_train    = 1/sqrt(2*SNR_train);

    % Different SNRs(in db) for which testing data is generated
    SNRdb       = 0:2:20; 
    SNR         = 10.^(SNRdb/10);
    stdv        = 1./sqrt(2*SNR);
    num_SNR_val = length(SNRdb);
end   

%% ========================== AWGN CHANNEL ================================
% =========================== OLD METHOD ==================================

%% NO USER(NU) CASE [63.39% of the time]

num_classes_NU         = 1 ;      % Number of classes for this case
num_train_perclass_NU  = 200000;  % Number of train data samples per class
num_test_perSNRval_NU  = 25000;   % Number of test data samples per SNR value
num_test_perclass_NU   = num_test_perSNRval_NU*num_SNR_val; % Number of test data samples per class
Ctrain_NU              =  complex(zeros(48,num_classes_NU*num_train_perclass_NU));
Ctest_NU_a1            =  complex(zeros(48,num_classes_NU*num_test_perclass_NU));
if Num_antennas == 2
    Ctest_NU_a2        =  complex(zeros(48,num_classes_NU*num_test_perclass_NU));
end

if Noise
    rng(8); % For repeatability of random number generation
    % Generating train data samples corresponding to 'SNRdb_train'
    for i = 1:num_train_perclass_NU
        Ctrain_NU(:,i)   = awgn (complex(zeros(48,1)), SNRdb_train);
    end

    rng(1998); % For repeatability of random number generation
    % Generating test data samples corresponding to different SNR values 
    % for antenna 1
    for i = 1:num_SNR_val
        snrdb     = SNRdb(i);
        col_range = (i-1)*num_test_perSNRval_NU+1 : i*num_test_perSNRval_NU;
        Ctest_NU_a1(:,col_range)   = awgn (complex(zeros(48,num_test_perSNRval_NU)), snrdb);
    end

    if Num_antennas == 2
        rng(2021); % For repeatability of random number generation
        % Generating test data samples corresponding to different SNR values 
        % for antenna 2
        for i = 1:num_SNR_val
            snrdb     = SNRdb(i);
            col_range = (i-1)*num_test_perSNRval_NU+1 : i*num_test_perSNRval_NU;
            Ctest_NU_a2(:,col_range)   = awgn (complex(zeros(48,num_test_perSNRval_NU)), snrdb);
        end
    end
end

%% ONE USER(OU) CASE [29.26% of the time]

num_classes_OU         = NUM_ZC_SEQ ; % Number of classes for this user case
num_train_perclass_OU  = 1000;        % Number of train data samples per class
num_test_perSNRval_OU  = 500;         % Number of test data samples per SNR value
num_test_perclass_OU   = num_test_perSNRval_OU*num_SNR_val; % Number of test data samples per class
Ctrain_OU              = complex(zeros(48,num_classes_OU*num_train_perclass_OU));
Ctest_OU               = complex(zeros(48,num_classes_OU*num_test_perclass_OU));

% Generating train data samples corresponding to 'SNRdb_train'
for n = 1 : num_classes_OU
   preamb = preambs(:,n);
   for num = 1 : num_train_perclass_OU
       rng(1998+num + num_train_perclass_OU * (n-1));
       Ctrain_OU(:,num + num_train_perclass_OU * (n-1) ) = awgn(preamb,SNRdb_train);
   end 
end

% Generating test data samples corresponding to different SNR values
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       for num = 1 : num_test_perSNRval_OU
           rng(98+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           Ctest_OU(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = awgn(preamb,snrdb);
       end
   end   
end

%% TWO USER(TU) CASE [6.38% of the time]

num_classes_TU         = nchoosek(NUM_ZC_SEQ,2); % Number of classes for this user case (46^C_2)
num_train_perclass_TU  = 100 ;                   % Number of train data samples per class
num_test_perSNRval_TU  = 50 ;                    % Number of test data samples per SNR value
num_test_perclass_TU   = num_test_perSNRval_TU*num_SNR_val; % Number of test data samples per class
Ctrain_TU              = complex(zeros(48,num_classes_TU*num_train_perclass_TU));
Ctest_TU               = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
combinations           = nchoosek(1:NUM_ZC_SEQ,2); % Different possible combinations of two ZC sequences

% Generating train data samples corresponding to different SNR values
for n = 1 : num_classes_TU
   s1 = combinations(n,1);
   s2 = combinations(n,2);
   preamb = preambs(:,s1) + preambs(:,s2);
   for num = 1 : num_train_perclass_TU
       rng(1998+num + num_train_perclass_TU * (n-1));
       Ctrain_TU(:,num + num_train_perclass_TU * (n-1)) = awgn(preamb,SNRdb_train);
   end  
end

% Generating test data samples corresponding to different SNR values
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb = preambs(:,s1) + preambs(:,s2);
       for num = 1 : num_test_perSNRval_TU
           rng(98+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           Ctest_TU(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = awgn(preamb,snrdb);
       end
   end   
end

%% SAVING THE GENERATED DATASET

% Stacking real and imaginary parts 
% => Number of features of each data sample is 48*2 = 96

Train_NU = [real(Ctrain_NU);imag(Ctrain_NU)]';
Test_NU  = [real(Ctest_NU);imag(Ctest_NU)]';

Train_OU = [real(Ctrain_OU);imag(Ctrain_OU)]';
Test_OU  = [real(Ctest_OU);imag(Ctest_OU)]';

Train_TU = [real(Ctrain_TU);imag(Ctrain_TU)]';
Test_TU  = [real(Ctest_TU);imag(Ctest_TU)]';

csvwrite("Train_NU.csv",Train_NU);
csvwrite("Train_OU.csv",Train_OU);
csvwrite("Train_TU.csv",Train_TU);
save("Train_NU","Train_NU");
save("Train_OU","Train_OU");
save("Train_TU","Train_TU");

csvwrite("Test_NU.csv",Test_NU);
csvwrite("Test_OU.csv",Test_OU);
csvwrite("Test_TU.csv",Test_TU);
save("Test_NU","Test_NU");
save("Test_OU","Test_OU");
save("Test_TU","Test_TU");

%% ========================== AWGN CHANNEL ================================
% =========================== NEW METHOD ==================================

%% NO USER(NU) CASE [63.39% of the time]

num_classes_NU         = 1 ;      % Number of classes for this case
num_train_perclass_NU  = 300000;  % Number of train data samples per class
num_test_perSNRval_NU  = 60000;   % Number of test data samples per SNR value
num_test_perclass_NU   = num_test_perSNRval_NU*num_SNR_val; % Number of test data samples per class
Ctrain_NU              =  complex(zeros(48,num_classes_NU*num_train_perclass_NU));
Ctest_NU_a1            =  complex(zeros(48,num_classes_NU*num_test_perclass_NU));
if Num_antennas == 2
    Ctest_NU_a2        =  complex(zeros(48,num_classes_NU*num_test_perclass_NU));
end

if Noise

    % Generating train data samples corresponding to 'SNRdb_train' 
    for n = 1 : num_classes_NU
       for num = 1 : num_train_perclass_NU
          rng(1998+num + num_train_perclass_NU * (n-1) ); 
          Awgn  =  stdv_train*complex(randn(48,1),randn(48,1));     
          Ctrain_NU(:,num + num_train_perclass_NU * (n-1) ) = Awgn;
   
       end 
    end

    % Generating test data samples corresponding to different SNR values
    % for antenna 1
    for i = 1:num_SNR_val
        for n = 1 : num_classes_NU
           for num = 1 : num_test_perSNRval_NU
               rng(98+num + num_test_perSNRval_NU * (n-1) + num_test_perSNRval_NU * num_classes_NU * (i-1));
               Awgn    = stdv(i)*complex(randn(48,1),randn(48,1));
               Ctest_NU_a1(:,num + num_test_perSNRval_NU * (n-1) + num_test_perSNRval_NU * num_classes_NU * (i-1)) = Awgn;
           end
       end   
    end
    
    % Generating test data samples corresponding to different SNR values
    % for antenna 2
    for i = 1:num_SNR_val
        for n = 1 : num_classes_NU 
           for num = 1 : num_test_perSNRval_NU
               rng(21+num + num_test_perSNRval_NU * (n-1) + num_test_perSNRval_NU * num_classes_NU * (i-1));
               Awgn  = stdv(i)*complex(randn(48,1),randn(48,1));

               Ctest_NU_a2(:,num + num_test_perSNRval_NU * (n-1) + num_test_perSNRval_NU * num_classes_NU * (i-1)) = Awgn;
           end
       end   
    end
end

%% ONE USER(OU) CASE [29.26% of the time]

num_classes_OU         = NUM_ZC_SEQ ; % Number of classes for this user case
num_train_perclass_OU  = 1000;        % Number of train data samples per class
num_test_perSNRval_OU  = 500;         % Number of test data samples per SNR value
num_test_perclass_OU   = num_test_perSNRval_OU*num_SNR_val; % Number of test data samples per class
Ctrain_OU              = complex(zeros(48,num_classes_OU*num_train_perclass_OU));
Ctest_OU_a1            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));
Ctest_OU_a2            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));


% Generating train data samples corresponding to 'SNRdb_train' 
for n = 1 : num_classes_OU
   preamb = preambs(:,n); 
   for num = 1 : num_train_perclass_OU
       rng(1998+num + num_train_perclass_OU * (n-1)); 
       Awgn            = stdv_train*complex(randn(48,1),randn(48,1));
       Ctrain_OU(:,num + num_train_perclass_OU * (n-1) ) = preamb + Awgn;
   end 
end

% Generating test data samples corresponding to different SNR values
% for antenna 1
for i = 1:num_SNR_val
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       for num = 1 : num_test_perSNRval_OU
           rng(98+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           Awgn            = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_OU_a1(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb + Awgn;
       end
   end   
end

% Generating test data samples corresponding to different SNR values
% for antenna 2
for i = 1:num_SNR_val
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       for num = 1 : num_test_perSNRval_OU
           rng(21+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           Awgn            = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_OU_a2(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb + Awgn;
       end
   end   
end

%% TWO USER(TU) CASE [6.38% of the time]

num_classes_TU         = nchoosek(NUM_ZC_SEQ,2); % Number of classes for this user case (46^C_2)
num_train_perclass_TU  = 100 ;                   % Number of train data samples per class
num_test_perSNRval_TU  = 50 ;                    % Number of test data samples per SNR value
num_test_perclass_TU   = num_test_perSNRval_TU*num_SNR_val; % Number of test data samples per class
Ctrain_TU              = complex(zeros(48,num_classes_TU*num_train_perclass_TU));
Ctest_TU_a1            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
Ctest_TU_a2            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
combinations           = nchoosek(1:NUM_ZC_SEQ,2); % Different possible combinations of two ZC sequences


% Generating train data samples corresponding to 'SNRdb_train'
for n = 1 : num_classes_TU
   s1 = combinations(n,1);
   s2 = combinations(n,2);
   preamb1 = preambs(:,s1);
   preamb2 = preambs(:,s2);
   for num = 1 : num_train_perclass_TU
       rng(1998+num + num_train_perclass_TU * (n-1));
       preamb  = preamb1 + preamb2;
       Awgn    = stdv_train*complex(randn(48,1),randn(48,1));
       Ctrain_TU(:,num + num_train_perclass_TU * (n-1)) = preamb + Awgn;
   end  
end

% Generating test data samples corresponding to different SNR values
% for antenna 1
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb1 = preambs(:,s1);
       preamb2 = preambs(:,s2);
       for num = 1 : num_test_perSNRval_TU
           rng(98+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           preamb  = preamb1 + preamb2;
           Awgn    = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_TU_a1(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb + Awgn;
       end
   end   
end

% Generating test data samples corresponding to different SNR values
% for antenna 2
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb1 = preambs(:,s1);
       preamb2 = preambs(:,s2);
       for num = 1 : num_test_perSNRval_TU
           rng(21+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           preamb  = preamb1 + preamb2;
           Awgn    = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_TU_a2(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb + Awgn;
       end
   end   
end


%% SAVING THE GENERATED DATASET

% Stacking real and imaginary parts 
% => Number of features of each data sample is 48*2 = 96


Train_NU    = [real(Ctrain_NU);imag(Ctrain_NU)]';
Test_NU_a1  = [real(Ctest_NU_a1);imag(Ctest_NU_a1)]';
Test_NU_a2  = [real(Ctest_NU_a2);imag(Ctest_NU_a2)]';

Train_OU    = [real(Ctrain_OU);imag(Ctrain_OU)]';
Test_OU_a1  = [real(Ctest_OU_a1);imag(Ctest_OU_a1)]';
Test_OU_a2  = [real(Ctest_OU_a2);imag(Ctest_OU_a2)]';

Train_TU    = [real(Ctrain_TU);imag(Ctrain_TU)]';
Test_TU_a1  = [real(Ctest_TU_a1);imag(Ctest_TU_a1)]';
Test_TU_a2  = [real(Ctest_TU_a2);imag(Ctest_TU_a2)]';

csvwrite("Train_NU.csv",Train_NU);
csvwrite("Train_OU.csv",Train_OU);
csvwrite("Train_TU.csv",Train_TU);
save("Train_NU","Train_NU");
save("Train_OU","Train_OU");
save("Train_TU","Train_TU");

csvwrite("Test_NU_a1.csv",Test_NU_a1);
csvwrite("Test_NU_a2.csv",Test_NU_a2);
csvwrite("Test_OU_a1.csv",Test_OU_a1);
csvwrite("Test_OU_a2.csv",Test_OU_a2);
csvwrite("Test_TU_a1.csv",Test_TU_a1);
csvwrite("Test_TU_a2.csv",Test_TU_a2);
save("Test_NU_a1","Test_NU_a1");
save("Test_NU_a2","Test_NU_a2");
save("Test_OU_a1","Test_OU_a1");
save("Test_OU_a2","Test_OU_a2");
save("Test_TU_a1","Test_TU_a1");
save("Test_TU_a2","Test_TU_a2");

%% ================== RAYLEIGH FADING + AWGN CHANNEL ======================
% =========================================================================

%% NO USER(NU) CASE [63.39% of the time]
% same as dataset from AWGN channel 

%% ONE USER(OU) CASE [29.26% of the time]

num_classes_OU         = NUM_ZC_SEQ ; % Number of classes for this case
num_train_perclass_OU  = 5000;        % Number of train data samples per class
num_test_perSNRval_OU  = 500;         % Number of test data samples per SNR value
num_test_perclass_OU   = num_test_perSNRval_OU*num_SNR_val; % Number of test data samples per class
Ctrain_OU              = complex(zeros(48,num_classes_OU*num_train_perclass_OU));
Ctest_OU_a1            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));
Ctest_OU_a2            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));

% Generating train data samples corresponding to 'SNRdb_train' 
for n = 1 : num_classes_OU
   preamb = preambs(:,n);
   for num = 1 : num_train_perclass_OU
       rng(1998+num + num_train_perclass_OU * (n-1)); 
       h = 1/sqrt(2)*complex(randn,randn);
       % constraint the channel coefficient
       while abs(10*log10(abs(h)^2)) > 10
           h = 1/sqrt(2)*complex(randn,randn);
       end
       preamb_rayleigh = h * preamb;
       Awgn            = stdv_train*complex(randn(48,1),randn(48,1));
       Ctrain_OU(:,num + num_train_perclass_OU * (n-1) ) = preamb_rayleigh + Awgn;
   end 
end

% Generating test data samples corresponding to different SNR values
% for antenna 1
for i = 1:num_SNR_val
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       for num = 1 : num_test_perSNRval_OU
           rng(98+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           h = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h)^2)) > 10
              h = 1/sqrt(2)*complex(randn,randn);
           end
           preamb_rayleigh = h * preamb;
           Awgn            = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_OU_a1(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rayleigh + Awgn;
       end
   end   
end

% Generating test data samples corresponding to different SNR values
% for antenna 2
for i = 1:num_SNR_val
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       for num = 1 : num_test_perSNRval_OU
           rng(21+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           h = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h)^2)) > 10
             h = 1/sqrt(2)*complex(randn,randn);
           end
           preamb_rayleigh = h * preamb;
           Awgn            = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_OU_a2(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rayleigh + Awgn;
       end
   end   
end

%% TWO USER(TU) CASE [6.38% of the time]

num_classes_TU         = nchoosek(NUM_ZC_SEQ,2); % Number of classes for this case (46^C_2)
num_train_perclass_TU  = 500 ;                   % Number of train data samples per class
num_test_perSNRval_TU  = 50 ;                    % Number of test data samples per SNR value
num_test_perclass_TU   = num_test_perSNRval_TU*num_SNR_val; % Number of test data samples per class
Ctrain_TU              = complex(zeros(48,num_classes_TU*num_train_perclass_TU));
Ctest_TU_a1            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
Ctest_TU_a2            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
combinations           = nchoosek(1:NUM_ZC_SEQ,2); % Different possible combinations of two ZC sequences

% Generating train data samples corresponding to 'SNRdb_train'
for n = 1 : num_classes_TU
   s1 = combinations(n,1);
   s2 = combinations(n,2);
   preamb1 = preambs(:,s1);
   preamb2 = preambs(:,s2);
   for num = 1 : num_train_perclass_TU
       rng(1998+num + num_train_perclass_TU * (n-1));
       h1 = 1/sqrt(2)*complex(randn,randn);
       % constraint the channel coefficient
       while abs(10*log10(abs(h1)^2)) > 10
           h1 = 1/sqrt(2)*complex(randn,randn);
       end
       h2 = 1/sqrt(2)*complex(randn,randn);
       % constraint the channel coefficient
       while abs(10*log10(abs(h2)^2)) > 10
           h2 = 1/sqrt(2)*complex(randn,randn);
       end
       preamb1_rayleigh = h1 * preamb1;
       preamb2_rayleigh = h2 * preamb2;
       preamb_rayleigh  = preamb1_rayleigh + preamb2_rayleigh;
       Awgn             = stdv_train*complex(randn(48,1),randn(48,1));
       Ctrain_TU(:,num + num_train_perclass_TU * (n-1)) = preamb_rayleigh + Awgn;
   end  
end

% Generating test data samples corresponding to different SNR values
% for antenna 1
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb1 = preambs(:,s1);
       preamb2 = preambs(:,s2);
       for num = 1 : num_test_perSNRval_TU
           rng(98+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           h1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h1)^2)) > 10
               h1 = 1/sqrt(2)*complex(randn,randn);
           end
           h2 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h2)^2)) > 10
               h2 = 1/sqrt(2)*complex(randn,randn);
           end
           preamb1_rayleigh = h1 * preamb1;
           preamb2_rayleigh = h2 * preamb2;
           preamb_rayleigh  = preamb1_rayleigh + preamb2_rayleigh;
           Awgn             = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_TU_a1(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rayleigh + Awgn;
       end
   end   
end

% Generating test data samples corresponding to different SNR values
% for antenna 2
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb1 = preambs(:,s1);
       preamb2 = preambs(:,s2);
       for num = 1 : num_test_perSNRval_TU
           rng(21+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           h1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h1)^2)) > 10
               h1 = 1/sqrt(2)*complex(randn,randn);
           end
           h2 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h2)^2)) > 10
               h2 = 1/sqrt(2)*complex(randn,randn);
           end
           preamb1_rayleigh = h1 * preamb1;
           preamb2_rayleigh = h2 * preamb2;
           preamb_rayleigh  = preamb1_rayleigh + preamb2_rayleigh;
           Awgn             = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_TU_a2(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rayleigh + Awgn;
       end
   end   
end

%% TWO USER(TU) CASE COLLISION DATA [6.38% of the time]

num_classes_TUC         = NUM_ZC_SEQ;
num_test_perSNRval_TUC  = 500 ;                    % Number of test data samples per SNR value
num_test_perclass_TUC   = num_test_perSNRval_TUC*num_SNR_val; % Number of test data samples per class
Ctest_TUC_a1             = complex(zeros(48,num_classes_TUC*num_test_perclass_TUC));
Ctest_TUC_a2             = complex(zeros(48,num_classes_TUC*num_test_perclass_TUC));

% Generating test data samples corresponding to different SNR values
% for antenna 1
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TUC 
       preamb = preambs(:,n);
       for num = 1 : num_test_perSNRval_TUC
           rng(98+num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1));
           h1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h1)^2)) > 10
               h1 = 1/sqrt(2)*complex(randn,randn);
           end
           h2 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h2)^2)) > 10
               h2 = 1/sqrt(2)*complex(randn,randn);
           end
           preamb_rayleigh1 = h1 * preamb;
           preamb_rayleigh2 = h2 * preamb;
           preamb_rayleigh  = preamb_rayleigh1 + preamb_rayleigh2;
           Awgn             = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_TUC_a1(:,num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1)) = preamb_rayleigh + Awgn;
       end
   end   
end

% Generating test data samples corresponding to different SNR values
% for antenna 2
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TUC 
       preamb = preambs(:,n);
       for num = 1 : num_test_perSNRval_TUC
           rng(21+num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1));
           h1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h1)^2)) > 10
               h1 = 1/sqrt(2)*complex(randn,randn);
           end
           h2 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h2)^2)) > 10
               h2 = 1/sqrt(2)*complex(randn,randn);
           end
           preamb_rayleigh1 = h1 * preamb;
           preamb_rayleigh2 = h2 * preamb;
           preamb_rayleigh  = preamb_rayleigh1 + preamb_rayleigh2;
           Awgn             = stdv(i)*complex(randn(48,1),randn(48,1));
           Ctest_TUC_a2(:,num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1)) = preamb_rayleigh + Awgn;
       end
   end   
end

%% SAVING THE GENERATED DATASET

% Stacking real and imaginary parts 
% => Number of features of each data sample is 48*2 = 96

Test_NU_a1  = [real(Ctest_NU_a1);imag(Ctest_NU_a1)]';
Test_NU_a2  = [real(Ctest_NU_a2);imag(Ctest_NU_a2)]';

Train_OU    = [real(Ctrain_OU);imag(Ctrain_OU)]';
Test_OU_a1  = [real(Ctest_OU_a1);imag(Ctest_OU_a1)]';
Test_OU_a2  = [real(Ctest_OU_a2);imag(Ctest_OU_a2)]';

Train_TU    = [real(Ctrain_TU);imag(Ctrain_TU)]';
Test_TU_a1  = [real(Ctest_TU_a1);imag(Ctest_TU_a1)]';
Test_TU_a2  = [real(Ctest_TU_a2);imag(Ctest_TU_a2)]';
Test_TUC_a1  = [real(Ctest_TUC_a1);imag(Ctest_TUC_a1)]';
Test_TUC_a2  = [real(Ctest_TUC_a2);imag(Ctest_TUC_a2)]';

csvwrite("Train_OU.csv",Train_OU);
csvwrite("Train_TU.csv",Train_TU);
save("Train_OU","Train_OU");
save("Train_TU","Train_TU");

csvwrite("Test_NU_a1.csv",Test_NU_a1);
csvwrite("Test_NU_a2.csv",Test_NU_a2);
csvwrite("Test_OU_a1.csv",Test_OU_a1);
csvwrite("Test_OU_a2.csv",Test_OU_a2);
csvwrite("Test_TU_a1.csv",Test_TU_a1);
csvwrite("Test_TU_a2.csv",Test_TU_a2);
csvwrite("Test_TUC_a1.csv",Test_TUC_a1);
csvwrite("Test_TUC_a2.csv",Test_TUC_a2);
save("Test_NU_a1","Test_NU_a1");
save("Test_NU_a2","Test_NU_a2");
save("Test_OU_a1","Test_OU_a1");
save("Test_OU_a2","Test_OU_a2");
save("Test_TU_a1","Test_TU_a1");
save("Test_TU_a2","Test_TU_a2");
save("Test_TUC_a1","Test_TUC_a1");
save("Test_TUC_a2","Test_TUC_a2");

%% ========================= EPA + AWGN CHANNEL ===========================
% =========================================================================

%% NO USER(NU) CASE [63.39% of the time]
% same as dataset from AWGN channel 

%% ONE USER(OU) CASE [29.26% of the time]

num_classes_OU         = NUM_ZC_SEQ;  % Number of classes for this case
num_train_perclass_OU  = 5000;        % Number of train data samples per class
num_test_perSNRval_OU  = 500;         % Number of test data samples per SNR value
num_test_perclass_OU   = num_test_perSNRval_OU*num_SNR_val; % Number of test data samples per class
Ctrain_OU              = complex(zeros(48,num_classes_OU*num_train_perclass_OU));
Ctest_OU_a1            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));
Ctest_OU_a2            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));

% Generating train data samples corresponding to 'SNRdb_train' 
for n = 1 : num_classes_OU
   preamb      = preambs(:,n);
   preamb_ifft = ifft(preamb,N_fft);
   preamb_tx   = [preamb_ifft(end-CP_len + 1:end,1);preamb_ifft]; 
   for num = 1 : num_train_perclass_OU
       chcfg.InitTime       = rand();
       [preamb_ltefad,info] = lteFadingChannel(chcfg,[preamb_tx;zeros(25,1)]);
       preamb_ltefad        = preamb_ltefad(CP_len+info.ChannelFilterDelay+1:end,1);
       preamb_fft           = fft(preamb_ltefad,N_fft);
       rng(1998+num + num_train_perclass_OU * (n-1)); 
       Awgn       = stdv_train*complex(randn(48,1),randn(48,1));
       preamb_rx  = preamb_fft(1:48,1) + Awgn;
       Ctrain_OU(:,num + num_train_perclass_OU * (n-1) ) = preamb_rx;
       % For emprical snr calculation
       SP(num + num_train_perclass_OU * (n-1)) = norm(preamb_fft(1:48,1))^2;
       NP(num + num_train_perclass_OU * (n-1)) = norm(Awgn)^2;
   end 
end
Esnr_tr_OU = 10*log10(mean(SP)/mean(NP));

% Generating test data samples corresponding to different SNR values
% for antenna 1 and 2
for i = 1:num_SNR_val
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       preamb_ifft = ifft(preamb,N_fft);
       preamb_tx   = [preamb_ifft(end-CP_len + 1:end,1);preamb_ifft];
       for num = 1 : num_test_perSNRval_OU
           chcfg.InitTime       = rand();
           [preamb_ltefad,info] = lteFadingChannel(chcfg,[preamb_tx;zeros(25,1)]);
           preamb_ltefad1       = preamb_ltefad(CP_len+info.ChannelFilterDelay+1:end,1);
           preamb_fft1          = fft(preamb_ltefad1,N_fft);
           preamb_ltefad2       = preamb_ltefad(CP_len+info.ChannelFilterDelay+1:end,2);
           preamb_fft2          = fft(preamb_ltefad2,N_fft);
           rng(98+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           Awgn1       = stdv(i)*complex(randn(48,1),randn(48,1));
           rng(21+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           Awgn2       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx1  = preamb_fft1(1:48,1) + Awgn1;
           preamb_rx2  = preamb_fft2(1:48,1) + Awgn2;
           Ctest_OU_a1(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rx1;
           Ctest_OU_a2(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rx2;
           % For emprical snr calculation
           SP_a1(num + num_test_perSNRval_OU * (n-1)) = norm(preamb_fft1(1:48,1))^2;
           NP_a1(num + num_test_perSNRval_OU * (n-1)) = norm(Awgn1)^2;
           SP_a2(num + num_test_perSNRval_OU * (n-1)) = norm(preamb_fft2(1:48,1))^2;
           NP_a2(num + num_test_perSNRval_OU * (n-1)) = norm(Awgn2)^2;
       end
    end   
    Esnr_a1_OU(i) = 10*log10(mean(SP_a1)/mean(NP_a2));
    Esnr_a2_OU(i) = 10*log10(mean(SP_a2)/mean(NP_a2));
end

%% TWO USER(TU) CASE [6.38% of the time]
num_classes_TU         = nchoosek(NUM_ZC_SEQ,2); % Number of classes for this case (46^C_2)
num_train_perclass_TU  = 500 ;                   % Number of train data samples per class
num_test_perSNRval_TU  = 50 ;                    % Number of test data samples per SNR value
num_test_perclass_TU   = num_test_perSNRval_TU*num_SNR_val; % Number of test data samples per class
Ctrain_TU              = complex(zeros(48,num_classes_TU*num_train_perclass_TU));
Ctest_TU_a1            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
Ctest_TU_a2            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
combinations           = nchoosek(1:NUM_ZC_SEQ,2); % Different possible combinations of two ZC sequences

% Generating train data samples corresponding to 'SNRdb_train'
for n = 1 : num_classes_TU
   s1 = combinations(n,1);
   s2 = combinations(n,2);
   preamb1 = preambs(:,s1);
   preamb_ifft1 = ifft(preamb1,N_fft);
   preamb_tx1   = [preamb_ifft1(end-CP_len + 1:end,1);preamb_ifft1];
   preamb2 = preambs(:,s2);
   preamb_ifft2 = ifft(preamb2,N_fft);
   preamb_tx2   = [preamb_ifft2(end-CP_len + 1:end,1);preamb_ifft2];
   for num = 1 : num_train_perclass_TU
       chcfg.InitTime        = rand();
       [preamb_ltefad1,info] = lteFadingChannel(chcfg,[preamb_tx1;zeros(25,1)]);
       preamb_ltefad1        = preamb_ltefad1(CP_len+info.ChannelFilterDelay+1:end,1);
       chcfg.InitTime        = rand();
       [preamb_ltefad2,info] = lteFadingChannel(chcfg,[preamb_tx2;zeros(25,1)]);
       preamb_ltefad2        = preamb_ltefad2(CP_len+info.ChannelFilterDelay+1:end,1);
       preamb_ltefad         = preamb_ltefad1 + preamb_ltefad2;
       preamb_fft = fft(preamb_ltefad,N_fft);
       rng(98+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
       Awgn       = stdv_train*complex(randn(48,1),randn(48,1));
       preamb_rx  = preamb_fft(1:48,1) +  Awgn;
       Ctrain_TU(:,num + num_train_perclass_TU * (n-1)) = preamb_rx;
       % For emprical snr calculation
       preamb_fft1 = fft(preamb_ltefad1,N_fft);
       preamb_fft2 = fft(preamb_ltefad2,N_fft);
       SP_p1(num + num_train_perclass_TU * (n-1)) = norm(preamb_fft1(1:48,1))^2;
       SP_p2(num + num_train_perclass_TU * (n-1)) = norm(preamb_fft2(1:48,1))^2;
       NP(num + num_train_perclass_TU * (n-1))    = norm(Awgn)^2;
   end  
end
Esnr_p1_tr_TU = 10*log10(mean(SP_p1)/mean(NP));
Esnr_p2_tr_TU = 10*log10(mean(SP_p2)/mean(NP));

% Generating test data samples corresponding to different SNR values
% for antenna 1 and 2
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb1 = preambs(:,s1);
       preamb_ifft1 = ifft(preamb1,N_fft);
       preamb_tx1   = [preamb_ifft1(end-CP_len + 1:end,1);preamb_ifft1];
       preamb2 = preambs(:,s2);
       preamb_ifft2 = ifft(preamb2,N_fft);
       preamb_tx2   = [preamb_ifft2(end-CP_len + 1:end,1);preamb_ifft2];
       for num = 1 : num_test_perSNRval_TU
           chcfg.InitTime        = rand();
           [preamb_ltefad1,info] = lteFadingChannel(chcfg,[preamb_tx1;zeros(25,1)]);
           preamb_ltefad1        = preamb_ltefad1(CP_len+info.ChannelFilterDelay+1:end,:);
           chcfg.InitTime        = rand();
           [preamb_ltefad2,info] = lteFadingChannel(chcfg,[preamb_tx2;zeros(25,1)]);
           preamb_ltefad2        = preamb_ltefad2(CP_len+info.ChannelFilterDelay+1:end,:);
           preamb_ltefad_a1      = preamb_ltefad1(:,1) + preamb_ltefad2(:,1);
           preamb_fft_a1 = fft(preamb_ltefad_a1,N_fft);
           rng(98+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           Awgn_a1       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx_a1  = preamb_fft_a1(1:48,1) +  Awgn_a1;
           preamb_ltefad_a2       = preamb_ltefad1(:,2) + preamb_ltefad2(:,2);
           preamb_fft_a2 = fft(preamb_ltefad_a2,N_fft);
           rng(21+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           Awgn_a2       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx_a2  = preamb_fft_a2(1:48,1) +  Awgn_a2;
           Ctest_TU_a1(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rx_a1;
           Ctest_TU_a2(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rx_a2;
           % For emprical snr calculation
           preamb_fft1 = fft(preamb_ltefad1(:,1),N_fft);
           preamb_fft2 = fft(preamb_ltefad2(:,1),N_fft);
           SP_p1(num + num_test_perSNRval_TU * (n-1)) = norm(preamb_fft1(1:48,1))^2;
           SP_p2(num + num_test_perSNRval_TU * (n-1)) = norm(preamb_fft2(1:48,1))^2;
           NP(num + num_test_perSNRval_TU * (n-1))    = norm(Awgn_a1)^2;
       end
    end  
    Esnr_p1_a1_TU(i) = 10*log10(mean(SP_p1)/mean(NP));
    Esnr_p2_a1_TU(i) = 10*log10(mean(SP_p2)/mean(NP));
end

%% TWO USER(TU) CASE COLLISION DATA [6.38% of the time]

num_classes_TUC         = NUM_ZC_SEQ;
num_test_perSNRval_TUC  = 500 ;                    % Number of test data samples per SNR value
num_test_perclass_TUC   = num_test_perSNRval_TUC*num_SNR_val; % Number of test data samples per class
Ctest_TUC_a1            = complex(zeros(48,num_classes_TUC*num_test_perclass_TUC));
Ctest_TUC_a2            = complex(zeros(48,num_classes_TUC*num_test_perclass_TUC));


% Generating test data samples corresponding to different SNR values
% for antenna 1 and 2
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TUC
       preamb = preambs(:,n);
       preamb_ifft = ifft(preamb,N_fft);
       preamb_tx   = [preamb_ifft(end-CP_len + 1:end,1);preamb_ifft];
       for num = 1 : num_test_perSNRval_TUC
           chcfg.InitTime        = rand();
           [preamb_ltefad1,info] = lteFadingChannel(chcfg,[preamb_tx;zeros(25,1)]);
           preamb_ltefad1        = preamb_ltefad1(CP_len+info.ChannelFilterDelay+1:end,:);
           chcfg.InitTime        = rand();
           [preamb_ltefad2,info] = lteFadingChannel(chcfg,[preamb_tx;zeros(25,1)]);
           preamb_ltefad2        = preamb_ltefad2(CP_len+info.ChannelFilterDelay+1:end,:);
           preamb_ltefad_a1      = preamb_ltefad1(:,1) + preamb_ltefad2(:,1);
           preamb_fft_a1 = fft(preamb_ltefad_a1,N_fft);
           rng(98+num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1));
           Awgn_a1       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx_a1  = preamb_fft_a1(1:48,1) +  Awgn_a1;
           preamb_ltefad_a2       = preamb_ltefad1(:,2) + preamb_ltefad2(:,2);
           preamb_fft_a2 = fft(preamb_ltefad_a2,N_fft);
           rng(21+num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1));
           Awgn_a2       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx_a2  = preamb_fft_a2(1:48,1) +  Awgn_a2;
           Ctest_TUC_a1(:,num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1)) = preamb_rx_a1;
           Ctest_TUC_a2(:,num + num_test_perSNRval_TUC * (n-1) + num_test_perSNRval_TUC * num_classes_TUC * (i-1)) = preamb_rx_a2;
       end
    end  
end

%% SAVING THE GENERATED DATASET

% Stacking real and imaginary parts 
% => Number of features of each data sample is 48*2 = 96


Test_NU_a1  = [real(Ctest_NU_a1);imag(Ctest_NU_a1)]';
Test_NU_a2  = [real(Ctest_NU_a2);imag(Ctest_NU_a2)]';

Train_OU    = [real(Ctrain_OU);imag(Ctrain_OU)]';
Test_OU_a1  = [real(Ctest_OU_a1);imag(Ctest_OU_a1)]';
Test_OU_a2  = [real(Ctest_OU_a2);imag(Ctest_OU_a2)]';

Train_TU    = [real(Ctrain_TU);imag(Ctrain_TU)]';
Test_TU_a1  = [real(Ctest_TU_a1);imag(Ctest_TU_a1)]';
Test_TU_a2  = [real(Ctest_TU_a2);imag(Ctest_TU_a2)]';
Test_TUC_a1  = [real(Ctest_TUC_a1);imag(Ctest_TUC_a1)]';
Test_TUC_a2  = [real(Ctest_TUC_a2);imag(Ctest_TUC_a2)]';

csvwrite("Train_OU.csv",Train_OU);
csvwrite("Train_TU.csv",Train_TU);
save("Train_OU","Train_OU");
save("Train_TU","Train_TU");

csvwrite("Test_NU_a1.csv",Test_NU_a1);
csvwrite("Test_NU_a2.csv",Test_NU_a2);
csvwrite("Test_OU_a1.csv",Test_OU_a1);
csvwrite("Test_OU_a2.csv",Test_OU_a2);
csvwrite("Test_TU_a1.csv",Test_TU_a1);
csvwrite("Test_TU_a2.csv",Test_TU_a2);
csvwrite("Test_TUC_a1.csv",Test_TUC_a1);
csvwrite("Test_TUC_a2.csv",Test_TUC_a2);
save("Test_NU_a1","Test_NU_a1");
save("Test_NU_a2","Test_NU_a2");
save("Test_OU_a1","Test_OU_a1");
save("Test_OU_a2","Test_OU_a2");
save("Test_TU_a1","Test_TU_a1");
save("Test_TU_a2","Test_TU_a2");
save("Test_TUC_a1","Test_TUC_a1");
save("Test_TUC_a2","Test_TUC_a2");


%% ====== RAYLEIGH FADING + AWGN CHANNEL + TIMING&FREQUENCY OFFSET ========
% =========================================================================

%% NO USER(NU) CASE [63.39% of the time]
% same as dataset from AWGN channel 

%% ONE USER(OU) CASE [29.26% of the time]

num_classes_OU         = NUM_ZC_SEQ;  % Number of classes for this case
num_train_perclass_OU  = 5000;        % Number of train data samples per class
num_test_perSNRval_OU  = 1000;        % Number of test data samples per SNR value
num_test_perclass_OU   = num_test_perSNRval_OU*num_SNR_val; % Number of test data samples per class
Ctrain_OU              = complex(zeros(48,num_classes_OU*num_train_perclass_OU));
Ctest_OU_a1            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));
Ctest_OU_a2            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));

% Generating train data samples corresponding to 'SNRdb_train' 
for n = 1 : num_classes_OU
   preamb      = preambs(:,n);
   preamb_ifft = ifft(preamb,N_fft);
   preamb_tx   = [preamb_ifft(end-CP_len + 1:end,1);preamb_ifft]; 
   for num = 1 : num_train_perclass_OU
       rng(1998+num + num_train_perclass_OU * (n-1)); 
       h = 1/sqrt(2)*complex(randn,randn);
       % constraint the channel coefficient
       while abs(10*log10(abs(h)^2)) > 10
           h = 1/sqrt(2)*complex(randn,randn);
       end
       % Timing offset
       % Randomnly pick timing offset from [0 CP_len-1]
       TO           = randi([0 CP_len-1]);

       preamb_rayfad = h*preamb_tx;
       % Frequency offset
       % Randomnly pick frequency offset from [-FO_max FO_max] 
       FO         = randi([-FO_max FO_max]); 
       alpha      = (2*pi*FO)/chcfg.SamplingRate;
       FO_indices = (0:length(preamb_rayfad)-1)';
       preamb_FO    = preamb_rayfad.*exp(1i*alpha*FO_indices);
       preamb_TOFO  = [zeros(TO,1);preamb_FO(1:end-TO,1)];
       preamb_fft   = fft(preamb_TOFO(CP_len+1:CP_len+N_fft,1),N_fft);
       Awgn         = stdv_train*complex(randn(48,1),randn(48,1));
 
       preamb_rx  = preamb_fft(1:48,1) + Awgn;
       Ctrain_OU(:,num + num_train_perclass_OU * (n-1) ) = preamb_rx;
   end
end



% Generating test data samples corresponding to different SNR values
% for antenna 1 and 2
for i = 1:num_SNR_val
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       preamb_ifft = ifft(preamb,N_fft);
       preamb_tx   = [preamb_ifft(end-CP_len + 1:end,1);preamb_ifft];
       for num = 1 : num_test_perSNRval_OU
           rng(98+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           h1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h1)^2)) > 10
              h1 = 1/sqrt(2)*complex(randn,randn);
           end
           Awgn_a1   = stdv(i)*complex(randn(48,1),randn(48,1));
           
           rng(21+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           h2 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h2)^2)) > 10
              h2 = 1/sqrt(2)*complex(randn,randn);
           end
           Awgn_a2   = stdv(i)*complex(randn(48,1),randn(48,1));

           % Timing offset
           % Randomnly pick timing offset from [0 CP_len_test-1]
           TO        = randi([0 CP_len_test-1]);
           % Frequency offset
           % Randomnly pick frequency offset from [-FO_max FO_max] 
           FO         = randi([-FO_max FO_max]); 
           alpha      = (2*pi*FO)/chcfg.SamplingRate;
           FO_indices = (0:length(N_fft)-1)';
           preamb_rayfad1 = h1 * preamb_tx;
           preamb_rayfad2 = h2 * preamb_tx;
           
           preamb_FO_a1  = preamb_rayfad1.*exp(1i*alpha*FO_indices);
           preamb_FO_a2  = preamb_rayfad2.*exp(1i*alpha*FO_indices);
           preamb_TOFO_a1  = [zeros(TO,1);preamb_FO_a1(1:end-TO,1)];
           preamb_TOFO_a2  = [zeros(TO,1);preamb_FO_a2(1:end-TO,1)];
           preamb_fft_a1   = fft(preamb_TOFO_a1(CP_len+1:CP_len+N_fft,1),N_fft);
           preamb_fft_a2   = fft(preamb_TOFO_a2(CP_len+1:CP_len+N_fft,1),N_fft);
           preamb_rx_a1  = preamb_fft_a1(1:48,1) + Awgn_a1;
           preamb_rx_a2  = preamb_fft_a2(1:48,1) + Awgn_a2;
           Ctest_OU_a1(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rx_a1;
           Ctest_OU_a2(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rx_a2;
          
       end
    end   
end

%% TWO USER(TU) CASE [6.38% of the time]
num_classes_TU         = nchoosek(NUM_ZC_SEQ,2); % Number of classes for this case (46^C_2)
num_train_perclass_TU  = 500 ;                   % Number of train data samples per class
num_test_perSNRval_TU  = 100 ;                    % Number of test data samples per SNR value
num_test_perclass_TU   = num_test_perSNRval_TU*num_SNR_val; % Number of test data samples per class
Ctrain_TU              = complex(zeros(48,num_classes_TU*num_train_perclass_TU));
Ctest_TU_a1            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
Ctest_TU_a2            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
combinations           = nchoosek(1:NUM_ZC_SEQ,2); % Different possible combinations of two ZC sequences

% Generating train data samples corresponding to 'SNRdb_train'
for n = 1 : num_classes_TU
   s1 = combinations(n,1);
   s2 = combinations(n,2);
   preamb1 = preambs(:,s1);
   preamb_ifft1 = ifft(preamb1,N_fft);
   preamb_tx1   = [preamb_ifft1(end-CP_len + 1:end,1);preamb_ifft1];
   preamb2 = preambs(:,s2);
   preamb_ifft2 = ifft(preamb2,N_fft);
   preamb_tx2   = [preamb_ifft2(end-CP_len + 1:end,1);preamb_ifft2];
   for num = 1 : num_train_perclass_TU
       rng(1998+num + num_train_perclass_TU * (n-1));
       h1 = 1/sqrt(2)*complex(randn,randn);
       % constraint the channel coefficient
       while abs(10*log10(abs(h1)^2)) > 10
           h1 = 1/sqrt(2)*complex(randn,randn);
       end
       h2 = 1/sqrt(2)*complex(randn,randn);
       % constraint the channel coefficient
       while abs(10*log10(abs(h2)^2)) > 10
           h2 = 1/sqrt(2)*complex(randn,randn);
       end
       % Timing offset
       % Randomnly pick timing offset from [0 CP_len-1]
       TO_p1           = randi([0 CP_len-1]);
       TO_p2           = randi([0 CP_len-1]);
       preamb_rayfad1 = h1 * preamb_tx1;
       preamb_rayfad2 = h2 * preamb_tx2;
       % Frequency offset
       % Randomnly pick frequency offset from [-FO_max FO_max] 
       FO_p1         = randi([-FO_max FO_max]); 
       FO_p2         = randi([-FO_max FO_max]);
       alpha_p1      = (2*pi*FO_p1)/chcfg.SamplingRate;
       alpha_p2      = (2*pi*FO_p2)/chcfg.SamplingRate;
       FO_indices    = (0:length(preamb_rayfad1)-1)';
       preamb_FO_p1  = preamb_rayfad1.*exp(1i*alpha_p1*FO_indices);
       preamb_FO_p2  = preamb_rayfad2.*exp(1i*alpha_p2*FO_indices);
       preamb_TOFO_p1  = [zeros(TO_p1,1);preamb_FO_p1(1:end-TO_p1,1)];
       preamb_TOFO_p2  = [zeros(TO_p2,1);preamb_FO_p2(1:end-TO_p2,1)];
       preamb_TOFO     = preamb_TOFO_p1 + preamb_TOFO_p2;
       preamb_fft      = fft(preamb_TOFO(CP_len+1:CP_len+N_fft,1),N_fft);
       Awgn       = stdv_train*complex(randn(48,1),randn(48,1));

       preamb_rx  = preamb_fft(1:48,1) +  Awgn;
       Ctrain_TU(:,num + num_train_perclass_TU * (n-1)) = preamb_rx;
   end  
end

% Generating test data samples corresponding to different SNR values
% for antenna 1 and 2
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb1 = preambs(:,s1);
       preamb_ifft1 = ifft(preamb1,N_fft);
       preamb_tx1   = [preamb_ifft1(end-CP_len + 1:end,1);preamb_ifft1];
       preamb2 = preambs(:,s2);
       preamb_ifft2 = ifft(preamb2,N_fft);
       preamb_tx2   = [preamb_ifft2(end-CP_len + 1:end,1);preamb_ifft2];
       for num = 1 : num_test_perSNRval_TU
           rng(98+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           h1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h1)^2)) > 10
             h1 = 1/sqrt(2)*complex(randn,randn);
           end
           h2 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h2)^2)) > 10
             h2 = 1/sqrt(2)*complex(randn,randn);
           end
           Awgn_a1  = stdv(i)*complex(randn(48,1),randn(48,1));
           
           rng(21+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           h1_1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h1_1)^2)) > 10
             h1_1 = 1/sqrt(2)*complex(randn,randn);
           end
           h2_1 = 1/sqrt(2)*complex(randn,randn);
           % constraint the channel coefficient
           while abs(10*log10(abs(h2_1)^2)) > 10
             h2_1 = 1/sqrt(2)*complex(randn,randn);
           end
           Awgn_a2  = stdv(i)*complex(randn(48,1),randn(48,1));

           % Timing offset
           % Randomnly pick timing offset from [0 CP_len_test-1]
           TO_p1           = randi([0 CP_len_test-1]);
           TO_p2           = randi([0 CP_len_test-1]);
           preamb_rayfad1 = h1 * preamb_tx1;
           preamb_rayfad1_1 = h1_1 * preamb_tx1;
           preamb_rayfad2 = h2 * preamb_tx2;
           preamb_rayfad2_1 = h2_1 * preamb_tx2;
           % Frequency offset
           % Randomnly pick frequency offset from [-FO_max FO_max] 
           FO_p1         = randi([-FO_max FO_max]); 
           FO_p2         = randi([-FO_max FO_max]);
           alpha_p1      = (2*pi*FO_p1)/chcfg.SamplingRate;
           alpha_p2      = (2*pi*FO_p2)/chcfg.SamplingRate;
           FO_indices    = (0:length(preamb_rayfad1)-1)';
           preamb_FO_p1_a1  = preamb_rayfad1.*exp(1i*alpha_p1*FO_indices);
           preamb_FO_p1_a2  = preamb_rayfad1_1.*exp(1i*alpha_p1*FO_indices);
           preamb_FO_p2_a1  = preamb_rayfad2.*exp(1i*alpha_p2*FO_indices);
           preamb_FO_p2_a2  = preamb_rayfad2_1.*exp(1i*alpha_p2*FO_indices);
           preamb_TOFO_p1_a1  = [zeros(TO_p1,1);preamb_FO_p1_a1(1:end-TO_p1,1)];
           preamb_TOFO_p1_a2  = [zeros(TO_p1,1);preamb_FO_p1_a2(1:end-TO_p1,1)];
           preamb_TOFO_p2_a1  = [zeros(TO_p2,1);preamb_FO_p2_a1(1:end-TO_p2,1)];
           preamb_TOFO_p2_a2  = [zeros(TO_p2,1);preamb_FO_p2_a2(1:end-TO_p2,1)];
           preamb_TOFO_a1     = preamb_TOFO_p1_a1 + preamb_TOFO_p2_a1;
           preamb_TOFO_a2     = preamb_TOFO_p1_a2 + preamb_TOFO_p2_a2;
           preamb_fft_a1      = fft(preamb_TOFO_a1(CP_len+1:CP_len+N_fft,1),N_fft);
           preamb_fft_a2      = fft(preamb_TOFO_a2(CP_len+1:CP_len+N_fft,1),N_fft);
           preamb_rx_a1  = preamb_fft_a1(1:48,1) +  Awgn_a1;
           preamb_rx_a2  = preamb_fft_a2(1:48,1) +  Awgn_a2;

           Ctest_TU_a1(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rx_a1;
           Ctest_TU_a2(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rx_a2;

       end
    end  
end

%% SAVING THE GENERATED DATASET

% Stacking real and imaginary parts 
% => Number of features of each data sample is 48*2 = 96

Train_NU    = [real(Ctrain_NU);imag(Ctrain_NU)]';
Test_NU_a1  = [real(Ctest_NU_a1);imag(Ctest_NU_a1)]';
Test_NU_a2  = [real(Ctest_NU_a2);imag(Ctest_NU_a2)]';

% Train_OU    = [real(Ctrain_OU);imag(Ctrain_OU)]';
% Test_OU_a1  = [real(Ctest_OU_a1);imag(Ctest_OU_a1)]';
% Test_OU_a2  = [real(Ctest_OU_a2);imag(Ctest_OU_a2)]';

% Train_TU    = [real(Ctrain_TU);imag(Ctrain_TU)]';
% Test_TU_a1  = [real(Ctest_TU_a1);imag(Ctest_TU_a1)]';
% Test_TU_a2  = [real(Ctest_TU_a2);imag(Ctest_TU_a2)]';

% csvwrite("Train_NU.csv",Train_NU);
% csvwrite("Train_OU.csv",Train_OU);
% csvwrite("Train_TU.csv",Train_TU);
save("Train_NU","Train_NU");
% save("Train_OU","Train_OU");
% save("Train_TU","Train_TU");

% csvwrite("Test_NU_a1.csv",Test_NU_a1);
% csvwrite("Test_NU_a2.csv",Test_NU_a2);
% csvwrite("Test_OU_a1.csv",Test_OU_a1);
% csvwrite("Test_OU_a2.csv",Test_OU_a2);
% csvwrite("Test_TU_a1.csv",Test_TU_a1);
% csvwrite("Test_TU_a2.csv",Test_TU_a2);
save("Test_NU_a1","Test_NU_a1");
save("Test_NU_a2","Test_NU_a2");
% save("Test_OU_a1","Test_OU_a1");
% save("Test_OU_a2","Test_OU_a2");
% save("Test_TU_a1","Test_TU_a1");
% save("Test_TU_a2","Test_TU_a2");


%% =========== EPA + AWGN CHANNEL + TIMING&FREQUENCY OFFSET ===============
% =========================================================================

%% NO USER(NU) CASE [63.39% of the time]
% same as dataset from AWGN channel 

%% ONE USER(OU) CASE [29.26% of the time]

num_classes_OU         = NUM_ZC_SEQ;  % Number of classes for this case
num_train_perclass_OU  = 5000;        % Number of train data samples per class
num_test_perSNRval_OU  = 1000;         % Number of test data samples per SNR value
num_test_perclass_OU   = num_test_perSNRval_OU*num_SNR_val; % Number of test data samples per class
Ctrain_OU              = complex(zeros(48,num_classes_OU*num_train_perclass_OU));
Ctest_OU_a1            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));
Ctest_OU_a2            = complex(zeros(48,num_classes_OU*num_test_perclass_OU));

% Generating train data samples corresponding to 'SNRdb_train' 
for n = 1 : num_classes_OU
   preamb      = preambs(:,n);
   preamb_ifft = ifft(preamb,N_fft);
   preamb_tx   = [preamb_ifft(end-CP_len + 1:end,1);preamb_ifft]; 
   for num = 1 : num_train_perclass_OU
       chcfg.InitTime       = rand();
       [preamb_ltefad,info] = lteFadingChannel(chcfg,[preamb_tx;zeros(25,1)]);
       preamb_ltefad        = preamb_ltefad(:,1);
       
       % Frequency offset
       % Randomnly pick frequency offset from [-FO_max FO_max] 
       FO         = randi([-FO_max FO_max]); 
       alpha      = (2*pi*FO)/chcfg.SamplingRate;
       FO_indices = (0:length(preamb_ltefad)-1)';
       preamb_FO  = preamb_ltefad.*exp(1i*alpha*FO_indices);
       
       % Timing offset
       % Randomnly pick timing offset from [0 CP_len-1]
       TO           = randi([0 CP_len-1]);
       preamb_TOFO  = [zeros(TO,1);preamb_FO(1:end-TO,1)];
       preamb_fft   = fft(preamb_TOFO(CP_len+info.ChannelFilterDelay+1:CP_len+info.ChannelFilterDelay+N_fft,1),N_fft);
       
       rng(1998+num + num_train_perclass_OU * (n-1)); 
       Awgn       = stdv_train*complex(randn(48,1),randn(48,1));
       preamb_rx  = preamb_fft(1:48,1) + Awgn;
       Ctrain_OU(:,num + num_train_perclass_OU * (n-1) ) = preamb_rx;
       % For emprical snr calculation
       SP(num + num_train_perclass_OU * (n-1)) = norm(preamb_fft(1:48,1))^2;
       NP(num + num_train_perclass_OU * (n-1)) = norm(Awgn)^2;
   end 
end
Esnr_tr_OU = 10*log10(mean(SP)/mean(NP));


% Generating test data samples corresponding to different SNR values
% for antenna 1 and 2
for i = 1:num_SNR_val
    for n = 1 : num_classes_OU
       preamb = preambs(:,n);
       preamb_ifft = ifft(preamb,N_fft);
       preamb_tx   = [preamb_ifft(end-CP_len + 1:end,1);preamb_ifft];
       for num = 1 : num_test_perSNRval_OU
           chcfg.InitTime       = rand();
           [preamb_ltefad,info] = lteFadingChannel(chcfg,[preamb_tx;zeros(25,1)]);
           
           % Frequency offset
           % Randomnly pick frequency offset from [-FO_max FO_max] 
           FO         = randi([-FO_max FO_max]); 
           alpha      = (2*pi*FO)/chcfg.SamplingRate;
           FO_indices = (0:length(preamb_ltefad(:,1))-1)';
           preamb_FO_a1  = preamb_ltefad(:,1).*exp(1i*alpha*FO_indices);
           preamb_FO_a2  = preamb_ltefad(:,2).*exp(1i*alpha*FO_indices);

           % Timing offset
           % Randomnly pick timing offset from [0 CP_len_test-1]
           TO              = randi([0 CP_len_test-1]);
           preamb_TOFO_a1  = [zeros(TO,1);preamb_FO_a1(1:end-TO,1)];
           preamb_TOFO_a2  = [zeros(TO,1);preamb_FO_a2(1:end-TO,1)];
           preamb_fft_a1   = fft(preamb_TOFO_a1(CP_len+info.ChannelFilterDelay+1:CP_len+info.ChannelFilterDelay+N_fft,1),N_fft);
           preamb_fft_a2   = fft(preamb_TOFO_a2(CP_len+info.ChannelFilterDelay+1:CP_len+info.ChannelFilterDelay+N_fft,1),N_fft);

           rng(98+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           Awgn_a1       = stdv(i)*complex(randn(48,1),randn(48,1));
           rng(21+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
           Awgn_a2       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx_a1  = preamb_fft_a1(1:48,1) + Awgn_a1;
           preamb_rx_a2  = preamb_fft_a2(1:48,1) + Awgn_a2;
           Ctest_OU_a1(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rx_a1;
           Ctest_OU_a2(:,num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1)) = preamb_rx_a2;
           % For emprical snr calculation
           SP_a1(num + num_test_perSNRval_OU * (n-1)) = norm(preamb_fft_a1(1:48,1))^2;
           NP_a1(num + num_test_perSNRval_OU * (n-1)) = norm(Awgn_a1)^2;
           SP_a2(num + num_test_perSNRval_OU * (n-1)) = norm(preamb_fft_a2(1:48,1))^2;
           NP_a2(num + num_test_perSNRval_OU * (n-1)) = norm(Awgn_a2)^2;
       end
    end   
    Esnr_a1_OU(i) = 10*log10(mean(SP_a1)/mean(NP_a1));
    Esnr_a2_OU(i) = 10*log10(mean(SP_a2)/mean(NP_a2));
end

%% TWO USER(TU) CASE [6.38% of the time]
num_classes_TU         = nchoosek(NUM_ZC_SEQ,2); % Number of classes for this case (46^C_2)
num_train_perclass_TU  = 500 ;                   % Number of train data samples per class
num_test_perSNRval_TU  = 100 ;                    % Number of test data samples per SNR value
num_test_perclass_TU   = num_test_perSNRval_TU*num_SNR_val; % Number of test data samples per class
Ctrain_TU              = complex(zeros(48,num_classes_TU*num_train_perclass_TU));
Ctest_TU_a1            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
Ctest_TU_a2            = complex(zeros(48,num_classes_TU*num_test_perclass_TU));
combinations           = nchoosek(1:NUM_ZC_SEQ,2); % Different possible combinations of two ZC sequences

% Generating train data samples corresponding to 'SNRdb_train'
for n = 1 : num_classes_TU
   s1 = combinations(n,1);
   s2 = combinations(n,2);
   preamb1 = preambs(:,s1);
   preamb_ifft1 = ifft(preamb1,N_fft);
   preamb_tx1   = [preamb_ifft1(end-CP_len + 1:end,1);preamb_ifft1];
   preamb2 = preambs(:,s2);
   preamb_ifft2 = ifft(preamb2,N_fft);
   preamb_tx2   = [preamb_ifft2(end-CP_len + 1:end,1);preamb_ifft2];
   for num = 1 : num_train_perclass_TU
       chcfg.InitTime        = rand();
       [preamb_ltefad1,info] = lteFadingChannel(chcfg,[preamb_tx1;zeros(25,1)]);
       preamb_ltefad1        = preamb_ltefad1(:,1);
       [preamb_ltefad2,info] = lteFadingChannel(chcfg,[preamb_tx2;zeros(25,1)]);
       preamb_ltefad2        = preamb_ltefad2(:,1);
       
       % Frequency offset
       % Randomnly pick frequency offset from [-FO_max FO_max] 
       FO_p1         = randi([-FO_max FO_max]); 
       FO_p2         = randi([-FO_max FO_max]);
       alpha_p1      = (2*pi*FO_p1)/chcfg.SamplingRate;
       alpha_p2      = (2*pi*FO_p2)/chcfg.SamplingRate;
       FO_indices    = (0:length(preamb_ltefad1)-1)';
       preamb_FO_p1  = preamb_ltefad1.*exp(1i*alpha_p1*FO_indices);
       preamb_FO_p2  = preamb_ltefad2.*exp(1i*alpha_p2*FO_indices);

       % Timing offset
       % Randomnly pick timing offset from [0 CP_len-1]
       TO_p1           = randi([0 CP_len-1]);
       TO_p2           = randi([0 CP_len-1]);
       preamb_TOFO_p1  = [zeros(TO_p1,1);preamb_FO_p1(1:end-TO_p1,1)];
       preamb_TOFO_p2  = [zeros(TO_p2,1);preamb_FO_p2(1:end-TO_p2,1)];
       preamb_TOFO     = preamb_TOFO_p1 + preamb_TOFO_p2;
       preamb_fft      = fft(preamb_TOFO(CP_len+info.ChannelFilterDelay+1:CP_len+info.ChannelFilterDelay+N_fft,1),N_fft);
       
       rng(98+num + num_test_perSNRval_OU * (n-1) + num_test_perSNRval_OU * num_classes_OU * (i-1));
       Awgn       = stdv_train*complex(randn(48,1),randn(48,1));
       preamb_rx  = preamb_fft(1:48,1) +  Awgn;
       Ctrain_TU(:,num + num_train_perclass_TU * (n-1)) = preamb_rx;
       % For emprical snr calculation
       preamb_fft1 = fft(preamb_TOFO_p1,N_fft);
       preamb_fft2 = fft(preamb_TOFO_p2,N_fft);
       SP_p1(num + num_train_perclass_TU * (n-1)) = norm(preamb_fft1(1:48,1))^2;
       SP_p2(num + num_train_perclass_TU * (n-1)) = norm(preamb_fft2(1:48,1))^2;
       NP(num + num_train_perclass_TU * (n-1))    = norm(Awgn)^2;
   end  
end
Esnr_p1_tr_TU = 10*log10(mean(SP_p1)/mean(NP));
Esnr_p2_tr_TU = 10*log10(mean(SP_p2)/mean(NP));

% Generating test data samples corresponding to different SNR values
% for antenna 1 and 2
for i = 1:num_SNR_val
    snrdb  = SNRdb(i);
    for n = 1 : num_classes_TU
       s1 = combinations(n,1);
       s2 = combinations(n,2);
       preamb1 = preambs(:,s1);
       preamb_ifft1 = ifft(preamb1,N_fft);
       preamb_tx1   = [preamb_ifft1(end-CP_len + 1:end,1);preamb_ifft1];
       preamb2 = preambs(:,s2);
       preamb_ifft2 = ifft(preamb2,N_fft);
       preamb_tx2   = [preamb_ifft2(end-CP_len + 1:end,1);preamb_ifft2];
       for num = 1 : num_test_perSNRval_TU
           chcfg.InitTime        = rand();
           [preamb_ltefad1,info] = lteFadingChannel(chcfg,[preamb_tx1;zeros(25,1)]);
           [preamb_ltefad2,info] = lteFadingChannel(chcfg,[preamb_tx2;zeros(25,1)]);
           
           % Frequency offset
           % Randomnly pick frequency offset from [-FO_max FO_max] 
           FO_p1         = randi([-FO_max FO_max]); 
           FO_p2         = randi([-FO_max FO_max]);
           alpha_p1      = (2*pi*FO_p1)/chcfg.SamplingRate;
           alpha_p2      = (2*pi*FO_p2)/chcfg.SamplingRate;
           FO_indices    = (0:length(preamb_ltefad1)-1)';
           preamb_FO_p1_a1  = preamb_ltefad1(:,1).*exp(1i*alpha_p1*FO_indices);
           preamb_FO_p1_a2  = preamb_ltefad1(:,2).*exp(1i*alpha_p1*FO_indices);
           preamb_FO_p2_a1  = preamb_ltefad2(:,1).*exp(1i*alpha_p2*FO_indices);
           preamb_FO_p2_a2  = preamb_ltefad2(:,2).*exp(1i*alpha_p2*FO_indices);

           % Timing offset
           % Randomnly pick timing offset from [0 CP_len_test-1]
           TO_p1           = randi([0 CP_len_test-1]);
           TO_p2           = randi([0 CP_len_test-1]);
           preamb_TOFO_p1_a1  = [zeros(TO_p1,1);preamb_FO_p1_a1(1:end-TO_p1,1)];
           preamb_TOFO_p1_a2  = [zeros(TO_p1,1);preamb_FO_p1_a2(1:end-TO_p1,1)];
           preamb_TOFO_p2_a1  = [zeros(TO_p2,1);preamb_FO_p2_a1(1:end-TO_p2,1)];
           preamb_TOFO_p2_a2  = [zeros(TO_p2,1);preamb_FO_p2_a2(1:end-TO_p2,1)];
           preamb_TOFO_a1     = preamb_TOFO_p1_a1 + preamb_TOFO_p2_a1;
           preamb_TOFO_a2     = preamb_TOFO_p1_a2 + preamb_TOFO_p2_a2;
           preamb_fft_a1      = fft(preamb_TOFO_a1(CP_len+info.ChannelFilterDelay+1:CP_len+info.ChannelFilterDelay+N_fft,1),N_fft);
           preamb_fft_a2      = fft(preamb_TOFO_a2(CP_len+info.ChannelFilterDelay+1:CP_len+info.ChannelFilterDelay+N_fft,1),N_fft);
       
           rng(98+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           Awgn_a1       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx_a1  = preamb_fft_a1(1:48,1) +  Awgn_a1;
           rng(21+num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1));
           Awgn_a2       = stdv(i)*complex(randn(48,1),randn(48,1));
           preamb_rx_a2  = preamb_fft_a2(1:48,1) +  Awgn_a2;
           Ctest_TU_a1(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rx_a1;
           Ctest_TU_a2(:,num + num_test_perSNRval_TU * (n-1) + num_test_perSNRval_TU * num_classes_TU * (i-1)) = preamb_rx_a2;
           % For emprical snr calculation
           preamb_fft1 = fft(preamb_TOFO_p1_a1,N_fft);
           preamb_fft2 = fft(preamb_TOFO_p2_a1,N_fft);
           SP_a1(num + num_test_perSNRval_TU * (n-1)) = norm(preamb_fft_a1(1:48,1))^2;
           SP_a2(num + num_test_perSNRval_TU * (n-1)) = norm(preamb_fft_a2(1:48,1))^2;
           NP_a1(num + num_test_perSNRval_TU * (n-1))    = norm(Awgn_a1)^2;
           NP_a2(num + num_test_perSNRval_TU * (n-1))    = norm(Awgn_a2)^2;
       end
    end  
    Esnr_a1_TU(i) = 10*log10(mean(SP_a1)/mean(NP_a1));
    Esnr_a2_TU(i) = 10*log10(mean(SP_a2)/mean(NP_a1));
end

%% SAVING THE GENERATED DATASET

% Stacking real and imaginary parts 
% => Number of features of each data sample is 48*2 = 96

Train_OU    = [real(Ctrain_OU);imag(Ctrain_OU)]';
Test_OU_a1  = [real(Ctest_OU_a1);imag(Ctest_OU_a1)]';
Test_OU_a2  = [real(Ctest_OU_a2);imag(Ctest_OU_a2)]';

Train_TU    = [real(Ctrain_TU);imag(Ctrain_TU)]';
Test_TU_a1  = [real(Ctest_TU_a1);imag(Ctest_TU_a1)]';
Test_TU_a2  = [real(Ctest_TU_a2);imag(Ctest_TU_a2)]';

csvwrite("Train_OU.csv",Train_OU);
csvwrite("Train_TU.csv",Train_TU);
save("Train_OU","Train_OU");
save("Train_TU","Train_TU");

csvwrite("Test_OU_a1.csv",Test_OU_a1);
csvwrite("Test_OU_a2.csv",Test_OU_a2);
csvwrite("Test_TU_a1.csv",Test_TU_a1);
csvwrite("Test_TU_a2.csv",Test_TU_a2);
save("Test_OU_a1","Test_OU_a1");
save("Test_OU_a2","Test_OU_a2");
save("Test_TU_a1","Test_TU_a1");
save("Test_TU_a2","Test_TU_a2");





