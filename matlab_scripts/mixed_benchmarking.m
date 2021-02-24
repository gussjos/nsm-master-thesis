% the plots are composed of files that contain following
% informations:

% collection.N - number of frames of each trajectory
% collection.Deff - diffusivity of the trajectory
% collection.iOC - integrated optical contrast of the trajectory

%% benchmarking_1
clear;
close all

DD=[10,20,50];
iiOC=[ 1e-4,2e-4,5e-4];
figure
path = '/home/gustaf/nsm_data/mixed data benchmarks barbora/collection_D11.mat';
BasicColor = ["blue","red","green"]
%/Volumes/nsm-results/simulated_tests/velocity0_distance20_timesteps10000/particle_tracking/collection_D

load(path)
  ff=find(collection.N>40);
  collectionF.N=collection.N(ff);
  collectionF.iOC_mean=collection.iOC_mean(ff);
  collectionF.Deff_mean=collection.Deff_mean(ff);

for i=1:length(collectionF.iOC_mean)
    scatter(collectionF.iOC_mean(i),collectionF.Deff_mean(i),'MarkerFaceAlpha',collectionF.N(i)/max(collectionF.N),...
        'MarkerFaceColor',"blue",...
        'MarkerEdgeColor','none'); hold on
end
xlabel('iOC (\mum)')
ylabel('D (\mum^2/s)')



return

%% find means from fit

clear;
BasicColor = ["blue","red","green","yellow"]
%close all

DD=fliplr([10,20,50]);
iiOC=[1e-4,2e-4,5e-4];
iOC_lim=[-0.5e-4 2e-4];
D_lim=[30 70];
kk=0; figure
path = '/home/gustaf/nsm_data/mixed data benchmarks barbora/collection_D11.mat';
load(path)
%%
for iD=1:3
    D=DD(iD);
    iOC = iiOC(iD)
    load(path)
      ff=find(collection.N>40);
      collectionF.N=collection.N(ff);
      collectionF.iOC_mean=collection.iOC_mean(ff);
      collectionF.Deff_mean=collection.Deff_mean(ff);

     [iOC_edges, Deff_edges, iOC_edges2, Deff_edges2, iOC_hist, Deff_hist, iOCxDeff_hist] = weighted_histogram (collectionF,0.5e-5,0.5, 2, iOC_lim, D_lim);


     fitresult=fit(iOC_edges',iOC_hist','gauss1','TolFun',1e-9);
     disp(fitresult)

     %iOC_mean(iiiOC,iD)=fitresult.b1;
     %iOC_std(iiiOC,iD)=fitresult.c1/sqrt(2);

     fitresult=fit(Deff_edges',Deff_hist','gauss1','TolFun',1e-9);
     disp(fitresult)

     %Deff_mean(iiiOC,iD)=fitresult.b1;
     %Deff_std(iiiOC,iD)=fitresult.c1/sqrt(2);
end

%% Save
clc
save('iOC_means_non-mixed_data_barbora','iOC_mean')
save('iOC_stds_non-mixed_data_barbora','iOC_std')
save('D_means_non-mixed_data_barbora','Deff_mean')
save('D_stds_non-mixed_data_barbora','Deff_std')


%% benchmarking_2 and benchmarking_3

clear;
BasicColor = ["blue","red","green"]
%close all

DD=fliplr([10,20,50]);
iiOC=[0.75e-4,1e-4,2e-4,5e-4];
iOC_lim=[-0.5e-4 5.5e-4];
D_lim=[0 120];
kk=0; figure
for iD=1:3
    D=DD(iD);
     for iiiOC=1:3
         iOC=iiOC(iiiOC);
            path = '/home/gustaf/nsm_data/non mixed benchmarks barbora/collection_D'
            filename = strcat(path,num2str(D),'_OC',num2str(iOC),'_D11.mat')
            load(filename)
              ff=find(collection.N>40);
              collectionF.N=collection.N(ff);
              collectionF.iOC_mean=collection.iOC_mean(ff);
              collectionF.Deff_mean=collection.Deff_mean(ff);
         
         [iOC_edges, Deff_edges, iOC_edges2, Deff_edges2, iOC_hist, Deff_hist, iOCxDeff_hist] = weighted_histogram (collectionF,0.5e-5,0.5, 2, iOC_lim, D_lim);
         

         subplot(3,2,iD*2-1)
         bar(iOC_edges,iOC_hist,'FaceColor',BasicColor(iiiOC),'EdgeColor','none'); hold on
         fitresult=fit(iOC_edges',iOC_hist','gauss1','TolFun',1e-9);
         iOC_edges_fit=linspace(iOC_lim(1),iOC_lim(2),500);
         plot(iOC_edges_fit,fitresult.a1*exp(-((iOC_edges_fit-fitresult.b1)/fitresult.c1).^2),'Color',BasicColor(iiiOC),'LineWidth',2);
         title(strcat('D=',num2str(D),'\mum^2/s'))
         xlim(iOC_lim)
         xlabel('iOC (\mum)')
         ylabel('Normalized counts')
         iOC_mean(iiiOC,iD)=fitresult.b1;
         iOC_std(iiiOC,iD)=fitresult.c1/sqrt(2);
         
         
         subplot(3,2,iiiOC*2)
         bar(Deff_edges,Deff_hist,'FaceColor',BasicColor(iD),'EdgeColor','none'); hold on
         fitresult=fit(Deff_edges',Deff_hist','gauss1','TolFun',1e-9);
         Deff_edges_fit=linspace(D_lim(1),D_lim(2),500);
         plot(Deff_edges_fit,fitresult.a1*exp(-((Deff_edges_fit-fitresult.b1)/fitresult.c1).^2),'Color',BasicColor(iD),'LineWidth',2);
         title(strcat('iOC=',num2str(iOC),'\mum'))
         xlim(D_lim)
         xlabel('D (\mum^2/s)')
         ylabel('Normalized counts')
         Deff_mean(iiiOC,iD)=fitresult.b1;
         Deff_std(iiiOC,iD)=fitresult.c1/sqrt(2);
         
     end
end

%% Plot

figure1=figure;
axes1 = axes('Parent',figure1,...
    'Position',[0.599 0.539923954372624 0.224 0.309885931558935]);
hold(axes1,'on');
plot(iiOC,iOC_std);
xlabel('iOC (\mum)')
ylabel('std(iOC) (\mum)')
leg=[];
for i=1:length(DD)
    leg{i}=strcat('D=',num2str(DD(i)),'\mum^2/s');
end
legend(leg)

axes2 = axes('Parent',figure1,...
    'Position',[0.59934090909091 0.118057741621717 0.21965909090909 0.31540233442391]);
hold(axes2,'on');
plot(DD,Deff_std');
xlabel('D (\mum^2/s)')
ylabel('std(D) (\mum^2/s)')
leg=[];
for i=1:length(iiOC)
    leg{i}=strcat('iOC=',num2str(iiOC(i)),'\mum');
end
legend(leg)

axes3 = axes('Parent',figure1,'Position',[0.13 0.11 0.405 0.74361216730038]);
hold(axes3,'on');
plot(repmat(iiOC,3,1),repmat(DD',1,3),'o','MarkerFaceColor',[0.831372559070587 0.815686285495758 0.7843137383461],'MarkerEdgeColor','none')
errorbar(iOC_mean,Deff_mean,Deff_std/2,Deff_std/2,iOC_std/2,iOC_std/2,'.'); hold on
xlabel('iOC (\mum)')
ylabel('D (\mum^2/s)')
box on
set(axes3,'XGrid','on','YGrid','on');

