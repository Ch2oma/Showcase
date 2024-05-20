%%Convolution and correlation related tasks including a simple speech recognition algorithm that compares with Google TTS voice. It also includes a noise generator to see its effects and impact on the algorithm and its accuracy.
%{
%% Part 2.2:
n = 0:11;
xi = zeros(1, length(n)); 
xi([3,4,5,9,10,11]) = 1;
psi = ConvFUNC(xi,xi);
figure;
subplot(2,1,1);
stem(n, xi, 'filled');
title('\xi[n]');
ylabel('\xi[n]');
xlabel('n');
subplot(2,1,2);
stem(0:length(psi)-1, psi, 'filled');
title('Convolution Result \psi[n] = \xi[n] * \xi[n]');
xlabel('n');
ylabel('psi[n]');
%}



%% Part 3.1:
%{
Si = -5:0.25:5;
Tau= Si;
Xi = (Si >= -5) & (Si <= 5);
Eta = (Si >= -2.5) & (Si <= 2.5);


figure;

for i = 1:length(Psi)
    partpsi = Psi(1:i);
    partni = ni(1:i);

    subplot(2,2,1);
    plot(Si, Xi, 'r');
    title('\xi[n]');
    xlim([-6,6]);

    subplot(2,2,2);
    plot(Tau, Eta, 'g');
    title('\eta[n]');

    shift_eta = fliplr(Eta);
    shift_eta = circshift(shift_eta, i);

    subplot(2,2,3);
 
    plot(Tau, shift_eta, 'b');
    hold on;
    plot(Si, Xi, 'r');
    title('flipped \eta[n] and shifted');
    xlim([-6,6]);
    ylim([-2,2]);
    hold off;

    subplot(2,2,4);
    plot(partni, partpsi, 'LineWidth', 2);
    title('The animation of the Convolution');
    xlabel('Index');
    ylabel('Convolution Result');
    xlim([-11 11]);
    ylim([-6 26]);
    pause(0.1);

end
%}




%% Part 4.2:
%{
fs = 8192;
total_duration = 10; % 10 seconds in total
recObj = audiorecorder(fs, 16, 1);
%recording part
fprintf('Please pronounce all digits within 10 seconds...\n');
recordblocking(recObj, total_duration);
combined_audio = getaudiodata(recObj);
filename = 'MyAudio.flac';
audiowrite(filename, combined_audio, fs);
fprintf('Combined recording saved as %s\n', filename);
%}


%cross correlation part

%{
filename = 'TotalNumber.flac';
[Speech, fs] = audioread('MyAudio.flac'); % Replace 'MyID.flac' with the actual file name

if size(Speech, 2) > 1
    monovoice = Speech(:,1);
else
    monovoice = Speech;
end

VoiceTime = (0:length(monovoice)-1)/fs;


starter = 3; 
ender = 4.2;
Segn1 = monovoice(starter*fs:ender*fs);
tN1 = (0:length(Segn1)-1)/fs;

audiowrite('n1.flac', Segn1, fs);

[crosscorr, tlg] = xcorr(monovoice, Segn1);
timeCorr = tlg/fs;

figure;
subplot(2,1,1);
plot(VoiceTime, monovoice);
title('My Speech Signal');
ylabel('Amplitude');
xlabel('Time (s)');

subplot(2,1,2);
plot(tN1,Segn1);
title('n1 portion');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-2,2]);

figure;
subplot(3,1,1);
plot(timeCorr, abs(crosscorr));
title('|ψ[n]|');
ylabel('Correlation');
xlabel('Time (s)');

subplot(3,1,2);
plot(timeCorr, abs(crosscorr).^2);
title('|ψ[n]|^2');
ylabel('Correlation');
xlabel('Time (s)');

subplot(3,1,3);
plot(timeCorr, abs(crosscorr).^4);
title('|ψ[n]|^4');
ylabel('Correlation');
xlabel('Time (s)');
%}




%PART5.1
%1111111
%{
[audio_array, Fs] = audioread('TotalNumber.flac');
audio_len = length(audio_array);

%soundsc(audio_array, Fs);

% Determine the power of the audio signal
p_signal = sum(audio_array.^2) / audio_len;
fprintf('Power of Signal: %f\n', p_signal);

SNR_value = 0.001; % Default: 10. Repeat for 0.1 and 0.001
p_noise = p_signal / SNR_value;
fprintf('Power of Noise for SNR=10: %f\n', p_noise);

rng(5);
awgn = sqrt(p_noise) .* randn([audio_len, 1]);

noisy_audio = audio_array + awgn;

soundsc(noisy_audio, Fs);
%}

%{
%PART 5.2
%22222222

[audio_array, Fs] = audioread('TotalNumber.flac');
if size(audio_array, 2) > 1
    audio_array = mean(audio_array, 2);
end

[filter, Freqs] = audioread('0.flac');
disp(Freqs);
if size(filter, 2) > 1
    filter = mean(filter, 2);
end

SNRbegin = 0.01;
SNRfinal = 0.001;
SNRsteps = -0.001;

figure;
index = 1;

for SNR = SNRbegin:SNRsteps:SNRfinal
    p_signal = sum(audio_array.^2) / length(audio_array);
    
    p_noise = p_signal / SNR;
    
    rng(10);
    awgn = sqrt(p_noise) .* randn(size(audio_array));
    
    noisy_audio = audio_array + awgn;
    
    output = xcorr(noisy_audio, filter);
    
    subplot(5, 2, index);
    plot(output);
    title(sprintf('SNR = %.3f', SNR));
    
    index = index + 1;
end

sgtitle('Cross-correlation Results for decrementing SNR');

%}

