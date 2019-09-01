%% 输入
% ------------ INPUTS -------------------
clc
clear

%Input is 'all_beahv'/'all_mats'
thresh = 0.001;

clear PMAT_CR rest_1_mats i j k m 
%% 模型生成

% ---------------------------------------

no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred = zeros(no_sub,1);


fprintf('\n Thresh # %6.3f',thresh);
tt_info = struct('train_pos',[],'train_neg',[],'test_pos',[],'test_neg',[]);
result = struct('model',[]);
for leftout = 1:no_sub
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    % leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    train_behav = all_behav;
    train_behav(leftout) = [];
        
    % correlate all edges with behavior

    [r_mat,p_mat] = corr(train_vcts',train_behav,'Type','Spearman');
    
    all_r_mat(:,:,leftout) = reshape(r_mat,no_node,no_node);
    all_p_mat(:,:,leftout) = reshape(p_mat,no_node,no_node);
    
    % set threshold and define masks
    
    pos_mask = zeros(no_node,no_node);
    neg_mask = zeros(no_node,no_node);
    
    pos_edges = find(r_mat > 0 & p_mat < thresh);
    neg_edges = find(r_mat < 0 & p_mat < thresh);
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    all_pos_mask(:,:,leftout) = pos_mask;
    all_neg_mask(:,:,leftout) = neg_mask;
    % get sum of all edges in TRAIN subs (divide by 2 to control for the
    % fact that matrices are symmetric)
    
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:size(train_sumpos)
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
% responseScale = iqr(train_behav);    

result(leftout).classificationSVM = fitcsvm(...
    [train_sumpos,train_sumneg], ...
    train_behav, ...
    'KernelFunction', 'gauss', ...
    'KernelScale',0.85,...
    'PolynomialOrder',[],...
    'BoxConstraint', 1, ...
    'ClassNames', [0; 1],...
    'Standardize', true);
%     'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('ShowPlots',false,'MaxObjectiveEvaluations',60),...

    test_mat = all_mats(:,:,leftout);
%     test_behav = all_behav(leftout);
    test_sumpos = sum(sum(test_mat.*pos_mask))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask))/2;

behav_pred(leftout) =predict(result(leftout).classificationSVM,[test_sumpos,test_sumneg]);

tt_info(leftout).train_pos = train_sumpos;
tt_info(leftout).train_neg = train_sumneg;
tt_info(leftout).test_pos = test_sumpos;
tt_info(leftout).test_neg = test_sumneg;
% acc(leftout) =  behav_pred(leftout)==all_behav(leftout);

gscatter(train_sumpos,train_sumneg,train_behav,[0 0.4470 0.74101 ;0.8500 0.3250 0.0980],'.',18,'off');
hold on
switch behav_pred(leftout)+all_behav(leftout)
    case 0%叉号×代表错误预测 星号*代表正确预测
        plot(test_sumpos,test_sumneg,'*','Color',[0 0.4470 0.74101],'MarkerSize',15)
    case 1
        if behav_pred(leftout)==0 && all_behav(leftout)==1
            plot(test_sumpos,test_sumneg,'x','Color',[0.8500 0.3250 0.0980],'MarkerSize',16)
        else%颜色代表被试的真实标签
            plot(test_sumpos,test_sumneg,'x','Color',[0 0.4470 0.74101],'MarkerSize',16)
        end
    case 2
        plot(test_sumpos,test_sumneg,'*','Color',[0.8500 0.3250 0.0980],'MarkerSize',15)
end
hold off
m(leftout)=getframe;
end
% movie(m,1)

% compare predicted and observed scores

accuracy = sum(behav_pred == all_behav)/no_sub
plotconfusion(all_behav',behav_pred');

clear dataset115_CIAS_FD leftout neg_edges neg_mask no_node no_sub pos_edges pos_mask ss test_mat test_sumneg test_sumpos train_behav train_mats train_sumneg train_sumpos train_vcts
clear r_mat p_mat
