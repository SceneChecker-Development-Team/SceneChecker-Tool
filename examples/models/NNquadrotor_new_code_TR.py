import torch
from scipy.integrate import ode
import numpy as np
import polytope as pc
from typing import Optional, List, Tuple
import math
from src.Waypoint import Waypoint

class FFNNC(torch.nn.Module):
    def __init__(self, D_in=6, D_out=8):
        super(FFNNC, self).__init__()
        self.layer1 = torch.nn.Linear(D_in, 20)
        self.layer2 = torch.nn.Linear(20, 20)
        self.layer3 = torch.nn.Linear(20, D_out)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x


def func1(t, vars, u):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    bx = u[3]
    by = u[4]
    bz = u[5]
    sc = u[6]

    sc = -1 * sc

    vx = vars[6]
    vy = vars[7]
    vz = vars[8]

    dvx = 9.81 * np.sin(u1) / np.cos(u1)
    dvy = -9.81 * np.sin(u2) / np.cos(u2)


    tmp1 = dvx * math.cos(sc) - dvy * math.sin(sc)
    tmp2 = dvx * math.sin(sc) + dvy * math.cos(sc)
    dvx = tmp1
    dvy = tmp2


    dvz = u3 - 9.81
    dx = vx
    dy = vy
    dz = vz
    dref_x = bx
    dref_y = by
    dref_z = bz
    return [dref_x, dref_y, dref_z, dx, dy, dz, dvx, dvy, dvz]


def runModel(initalCondition, time_bound, time_step, ref_input):
    bias1 = [0.005067077465355396, 0.013770184479653835, 0.02960527129471302, -0.005076207220554352,
             0.00986097939312458, -0.004963981453329325, 0.005067163147032261, 0.022097138687968254,
             0.005067095160484314, -0.009861057624220848, -0.005051563028246164, -0.013770153746008873,
             0.009860980324447155, 0.009861058555543423, 0.013770055025815964, 0.005067105405032635,
             -0.01376994326710701, 0.005067072343081236, -0.005067067686468363, -0.013769976794719696]
    bias2 = [-0.009237067773938179, -0.00437132129445672, 0.0007612668559886515, 0.009431629441678524,
             -0.04935901612043381, 0.00892704352736473, 0.00891881249845028, -0.042356036603450775, 0.03377627208828926,
             0.014071502722799778, -0.0018434594385325909, -0.0006053714314475656, -0.0038432874716818333,
             -0.007012223359197378, -0.0007034881855361164, 0.007561248727142811, -0.042776428163051605,
             0.009373181499540806, 0.0031296780798584223, 0.008734943345189095]
    bias3 = [-0.8870829939842224, -1.152485728263855, -1.3024290800094604, -1.1338839530944824, -0.12526285648345947,
             -0.35318782925605774, -0.9353211522102356, -1.0099754333496094]
    weight1 = [
        [-0.012771551497280598, 0.8522672653198242, 0.005031325854361057, 0.010413140058517456, 0.5228086709976196,
         0.0038109351880848408],
        [0.01777890883386135, 0.011899493634700775, -0.8380187153816223, 0.010888529941439629, 0.011424682103097439,
         -0.5449933409690857],
        [0.01807912439107895, 0.012226282618939877, -0.8379306793212891, 0.011077952571213245, 0.011494635604321957,
         -0.5451062917709351],
        [0.01277169119566679, -0.8522672057151794, -0.005030765198171139, -0.010413102805614471, -0.5228087902069092,
         -0.0038107011932879686],
        [-0.8513970971107483, -0.01639718748629093, -0.015279111452400684, -0.5237646102905273, -0.014861795119941235,
         -0.008371188305318356],
        [0.012770231813192368, -0.8522675633430481, -0.005036665592342615, -0.010414107702672482, -0.5228080749511719,
         -0.0038130702450871468],
        [-0.012771555222570896, 0.8522672653198242, 0.005031300242990255, 0.01041315495967865, 0.5228086709976196,
         0.0038109151646494865],
        [-0.8513400554656982, -0.01752384752035141, -0.01597718894481659, -0.5237532258033752, -0.01596754789352417,
         -0.009223726578056812],
        [-0.012771563604474068, 0.8522672057151794, 0.005031331907957792, 0.010413155891001225, 0.5228087902069092,
         0.0038109072484076023],
        [0.8513972163200378, 0.016397155821323395, 0.015279067680239677, 0.5237646698951721, 0.014861810952425003,
         0.008371211588382721],
        [0.012771359644830227, -0.8522672653198242, -0.005032065790146589, -0.010413287207484245, -0.5228086709976196,
         -0.003811270697042346],
        [-0.01777891255915165, -0.011899505741894245, 0.8380184173583984, -0.010888525284826756, -0.011424724943935871,
         0.5449937582015991],
        [-0.8513973355293274, -0.016397180035710335, -0.01527908630669117, -0.5237643718719482, -0.014861813746392727,
         -0.008371248841285706],
        [-0.8513972163200378, -0.01639718748629093, -0.0152790741994977, -0.5237645506858826, -0.01486178208142519,
         -0.008371269330382347],
        [0.01777891255915165, 0.011899520643055439, -0.8380185961723328, 0.010888495482504368, 0.011424724012613297,
         -0.54499351978302],
        [-0.012771572917699814, 0.8522672653198242, 0.005031317938119173, 0.010413181968033314, 0.5228086709976196,
         0.003810951951891184],
        [-0.017778869718313217, -0.011899495497345924, 0.838018536567688, -0.010888428427278996, -0.011424727737903595,
         0.54499351978302],
        [-0.012771587818861008, 0.8522674441337585, 0.005031261593103409, 0.010413181968033314, 0.5228084325790405,
         0.0038109596353024244],
        [0.012771571055054665, -0.8522672057151794, -0.00503126997500658, -0.010413173586130142, -0.5228086113929749,
         -0.0038109286688268185],
        [-0.017778851091861725, -0.011899485252797604, 0.838018536567688, -0.010888493619859219, -0.0114247165620327,
         0.54499351978302]]
    weight2 = [
        [0.004485692363232374, 0.4092908799648285, 0.40233033895492554, -0.0044856867752969265, -0.008494298905134201,
         -0.004485597368329763, 0.004485687240958214, -0.008073708973824978, 0.004485689103603363, 0.008494298905134201,
         -0.004485669545829296, -0.4092909097671509, -0.00849429052323103, -0.008494291454553604, 0.4092908799648285,
         0.004485682118684053, -0.4092908799648285, 0.004485685378313065, -0.004485683515667915, -0.4092908799648285],
        [-0.0012539782328531146, 0.011227603070437908, 0.010983745567500591, 0.0012537383008748293,
         -0.44823628664016724, 0.0012566098012030125, -0.0012539738090708852, -0.4422348737716675,
         -0.0012539689196273685, 0.44823625683784485, 0.0012543690390884876, -0.011227612383663654, -0.4482363164424896,
         -0.4482361674308777, 0.011227606795728207, -0.001253972644917667, -0.011227593757212162, -0.001253975322470069,
         0.001253973226994276, -0.01122759748250246],
        [0.0007343984907492995, -0.011749573983252048, -0.011509227566421032, -0.0007341671735048294,
         0.44823724031448364, -0.0007370298844762146, 0.0007343830075114965, 0.4421602189540863, 0.0007343870238400996,
         -0.44823718070983887, -0.0007348008803091943, 0.011749575845897198, 0.44823724031448364, 0.44823721051216125,
         -0.011749569326639175, 0.0007343803299590945, 0.011749548837542534, 0.0007343843462876976,
         -0.0007343952311202884, 0.011749545112252235],
        [0.0032850054558366537, 0.012248189188539982, 0.012028064578771591, -0.003285229206085205, -0.4482342302799225,
         -0.0032824152149260044, 0.003285015933215618, -0.44198647141456604, 0.0032850138377398252, 0.44823431968688965,
         -0.0032846173271536827, -0.012248193845152855, -0.4482342004776001, -0.4482342600822449, 0.01224818080663681,
         0.0032850129064172506, -0.012248178012669086, 0.003285012673586607, -0.003285010578110814,
         -0.012248177081346512],
        [-0.33326631784439087, 0.001286709913983941, 0.0012743606930598617, 0.33326631784439087, 0.008707520551979542,
         0.3332684338092804, -0.33326634764671326, 0.009328183718025684, -0.33326634764671326, -0.008707527071237564,
         0.33326682448387146, -0.001286706537939608, 0.008707529865205288, 0.008707528933882713, 0.0012867064215242863,
         -0.33326634764671326, -0.001286699203774333, -0.3332662582397461, 0.3332662880420685, -0.0012867129407823086],
        [-0.005634149070829153, -0.40926215052604675, -0.4023561179637909, 0.005634183995425701, 0.008391384966671467,
         0.005634055007249117, -0.005634150467813015, 0.007969524711370468, -0.005634148605167866,
         -0.008391385897994041, 0.005634150002151728, 0.4092622399330139, 0.008391381241381168, 0.008391374722123146,
         -0.40926244854927063, -0.0056341588497161865, 0.40926244854927063, -0.0056341467425227165,
         0.005634146276861429, 0.409262478351593],
        [-0.006210127845406532, -0.4092416763305664, -0.4023682475090027, 0.006210171617567539, 0.00855502113699913,
         0.006210030987858772, -0.006210129242390394, 0.008134675212204456, -0.006210132036358118,
         -0.008555027656257153, 0.006210120394825935, 0.4092416763305664, 0.008555008098483086, 0.008555013686418533,
         -0.4092416763305664, -0.006210125517100096, 0.409241646528244, -0.00621012831106782, 0.006210136227309704,
         0.4092416763305664],
        [0.3332020938396454, 0.0034835836850106716, 0.0034986836835741997, -0.3332015573978424, -0.011808326467871666,
         -0.3332063853740692, 0.3332020342350006, -0.012397287413477898, 0.3332020342350006, 0.01180832739919424,
         -0.3332027792930603, -0.0034835890401154757, -0.011808333918452263, -0.01180832739919424,
         0.0034835883416235447, 0.3332020342350006, -0.0034835846163332462, 0.3332020342350006, -0.333202064037323,
         -0.003483586013317108],
        [0.3332599997520447, -0.00046248571015894413, -0.0004488642734941095, -0.3332599699497223,
         -0.009213998913764954, -0.3332628607749939, 0.3332599997520447, -0.009829509072005749, 0.3332599997520447,
         0.009213999845087528, -0.33326059579849243, 0.00046249059960246086, -0.009214004501700401,
         -0.009213997051119804, -0.0004624773282557726, 0.3332599997520447, 0.0004624743014574051, 0.3332599997520447,
         -0.3332599997520447, 0.00046248442959040403],
        [0.005426416639238596, 0.012804088182747364, 0.012594926171004772, -0.005426647607237101, -0.44819632172584534,
         -0.005423834081739187, 0.005426429677754641, -0.44185635447502136, 0.005426422227174044, 0.4481962323188782,
         -0.005426030606031418, -0.012804095633327961, -0.4481962323188782, -0.4481961727142334, 0.01280409935861826,
         0.005426430609077215, -0.012804084457457066, 0.00542643154039979, -0.005426420830190182,
         -0.012804084457457066],
        [-8.470881584798917e-05, -0.01166805811226368, -0.011431642808020115, 8.493004861520603e-05,
         0.44824597239494324, 8.206536585930735e-05, -8.470812463201582e-05, 0.4421427845954895, -8.470044849673286e-05,
         -0.4482460021972656, 8.428884029854089e-05, 0.011668059974908829, 0.4482460021972656, 0.44824597239494324,
         -0.011668050661683083, -8.470762259094045e-05, 0.011668048799037933, -8.469591557513922e-05,
         8.470559259876609e-05, 0.011668049730360508],
        [0.0007104332908056676, -0.012103226035833359, -0.011863806284964085, -0.0007102005183696747,
         0.4482336640357971, -0.0007130496669560671, 0.0007104334654286504, 0.4421180486679077, 0.0007104235119186342,
         -0.44823363423347473, -0.0007108247373253107, 0.012103239074349403, 0.4482337534427643, 0.44823357462882996,
         -0.012103230692446232, 0.0007104267715476453, 0.012103220447897911, 0.0007104279939085245,
         -0.0007104240357875824, 0.012103220447897911],
        [-0.0008108518086373806, -0.011628665030002594, -0.011395211331546307, 0.0008111011702567339,
         0.4482516348361969, 0.0008082437561824918, -0.0008108714246191084, 0.44211941957473755, -0.000810873054433614,
         -0.4482516050338745, 0.0008104708977043629, 0.011628672480583191, 0.4482516050338745, 0.4482516348361969,
         -0.011628661304712296, -0.0008108695619739592, 0.011628646403551102, -0.0008108615875244141,
         0.0008108713664114475, 0.011628646403551102],
        [0.0052762399427592754, 0.4092657268047333, 0.4023624062538147, -0.005276260897517204, -0.008586729876697063,
         -0.0052761598490178585, 0.005276238080114126, -0.008165040984749794, 0.005276246462017298,
         0.008586726151406765, -0.0052762338891625404, -0.4092656970024109, -0.00858672522008419, -0.008586717769503593,
         0.4092656970024109, 0.005276253912597895, -0.40926575660705566, 0.005276249721646309, -0.005276253912597895,
         -0.4092658758163452],
        [0.0004666325112339109, -0.01200425997376442, -0.011765711009502411, -0.00046640209620818496,
         0.44823700189590454, -0.00046925980132073164, 0.00046662817476317286, 0.4421238601207733,
         0.0004666206077672541, -0.448236882686615, -0.00046703135012649, 0.012004264630377293, 0.44823694229125977,
         0.4482368230819702, -0.01200425811111927, 0.0004666294262278825, 0.012004253454506397, 0.0004666416789405048,
         -0.00046663961256854236, 0.012004251591861248],
        [-0.006058018188923597, -0.40924715995788574, -0.4023823142051697, 0.006058032624423504, 0.008350761607289314,
         0.006057909224182367, -0.006058020517230034, 0.007928609848022461, -0.00605802284553647, -0.00835077092051506,
         0.006058006081730127, 0.40924718976020813, 0.00835077092051506, 0.00835077092051506, -0.40924715995788574,
         -0.006058022379875183, 0.4092472195625305, -0.006058013532310724, 0.0060580214485526085, 0.40924715995788574],
        [0.3332037329673767, 0.0033146513160318136, 0.003330634441226721, -0.33320334553718567, -0.011781765148043633,
         -0.3332081437110901, 0.33320361375808716, -0.012370237149298191, 0.3332037031650543, 0.011781767010688782,
         -0.33320438861846924, -0.0033146499190479517, -0.011781766079366207, -0.011781767010688782,
         0.003314657835289836, 0.3332037329673767, -0.0033146559726446867, 0.33320367336273193, -0.3332037329673767,
         -0.0033146515488624573],
        [-0.005700466223061085, -0.4092574417591095, -0.40236255526542664, 0.005700497422367334, 0.008488606661558151,
         0.005700360052287579, -0.005700466688722372, 0.008067479357123375, -0.00570047739893198, -0.008488614112138748,
         0.005700456909835339, 0.4092574417591095, 0.008488607592880726, 0.008488602004945278, -0.4092575013637543,
         -0.005700466223061085, 0.4092575013637543, -0.005700462963432074, 0.005700469017028809, 0.4092574715614319],
        [0.0007849707617424428, 0.012151014059782028, 0.011918344534933567, -0.0007852140697650611, -0.4482424557209015,
         -0.000782380870077759, 0.0007849783287383616, -0.4420732855796814, 0.0007849846151657403, 0.4482424259185791,
         -0.0007845927611924708, -0.012151009403169155, -0.4482423961162567, -0.4482423961162567, 0.012150995433330536,
         0.0007849778048694134, -0.012150992639362812, 0.0007849839166738093, -0.0007849822868593037,
         -0.012150988914072514],
        [-0.005965784657746553, -0.40925177931785583, -0.4023708701133728, 0.005965803749859333, 0.008349047973752022,
         0.005965673830360174, -0.005965787451714277, 0.0079267006367445, -0.005965782329440117, -0.008349056355655193,
         0.0059657711535692215, 0.4092518091201782, 0.008349052630364895, 0.008349040523171425, -0.4092518091201782,
         -0.00596578698605299, 0.40925195813179016, -0.00596578698605299, 0.005965790245682001, 0.4092518985271454]]
    weight3 = [
        [-335.4917907714844, 295.4498596191406, -300.9907531738281, 292.68359375, 439.89520263671875, 340.9864807128906,
         341.3768615722656, -446.5185852050781, -442.7583312988281, 283.7157287597656, -300.0808410644531,
         -300.40203857421875, -298.2679748535156, -338.4247741699219, -298.95599365234375, 342.16082763671875,
         -442.63568115234375, 340.7982482910156, 291.43646240234375, 340.5745544433594],
        [333.4602355957031, 283.6884460449219, -290.77532958984375, 284.6348571777344, 458.39697265625,
         -331.57843017578125, -330.1046142578125, -458.7938232421875, -459.3420104980469, 277.3214416503906,
         -290.0921936035156, -291.56329345703125, -287.3941955566406, 332.5392150878906, -289.87408447265625,
         -327.3385009765625, -456.78961181640625, -331.62420654296875, 282.5384521484375, -330.56890869140625],
        [-337.1375427246094, 289.10272216796875, -294.45098876953125, 296.2870788574219, -438.8525085449219,
         339.5340881347656, 337.7414855957031, 435.8607177734375, 437.567138671875, 293.1355895996094,
         -295.7801208496094, -293.8027038574219, -295.40887451171875, -337.87371826171875, -293.3611755371094,
         339.5439147949219, 437.2536315917969, 338.9185791015625, 289.39752197265625, 338.4055480957031],
        [331.2817687988281, 278.56243896484375, -286.8565979003906, 290.38287353515625, -383.39654541015625,
         -332.671630859375, -332.4136047363281, 458.2372741699219, 394.3974609375, 288.79791259765625,
         -287.97247314453125, -287.65728759765625, -287.8821105957031, 333.19647216796875, -286.55657958984375,
         -330.1717834472656, 458.9079895019531, -332.5555114746094, 282.8378601074219, -332.8061828613281],
        [-327.96527099609375, -282.6752014160156, 281.63214111328125, -289.4588928222656, 376.37738037109375,
         333.9535217285156, 335.0184020996094, -457.1034851074219, -390.4696960449219, -294.4223327636719,
         281.4463806152344, 281.6940002441406, 284.7673645019531, -332.4997253417969, 284.1268615722656,
         335.67401123046875, -454.1480407714844, 334.2760009765625, -288.58868408203125, 333.94891357421875],
        [341.65960693359375, -295.7402648925781, 292.73046875, -298.5085754394531, 437.5282287597656,
         -340.9335021972656, -338.32208251953125, -443.3055114746094, -439.6506042480469, -301.7493591308594,
         292.2878723144531, 291.4482116699219, 295.3114013671875, 341.15484619140625, 294.27960205078125,
         -337.8680419921875, -440.09320068359375, -340.21820068359375, -298.2076721191406, -340.53692626953125],
        [-330.5609130859375, -291.5669860839844, 287.8730773925781, -286.07989501953125, -456.10418701171875,
         332.3810119628906, 332.67425537109375, 451.7678527832031, 455.01104736328125, -287.4740905761719,
         285.10052490234375, 287.6368103027344, 286.7076721191406, -332.6780090332031, 289.9099426269531,
         333.3375549316406, 453.9902648925781, 332.9720764160156, -292.4762878417969, 331.2679748535156],
        [340.7230224609375, -301.74951171875, 296.9709777832031, -292.6571960449219, -440.14434814453125,
         -343.39404296875, -343.3323059082031, 441.12506103515625, 440.0010070800781, -290.7984924316406,
         294.84613037109375, 295.7033996582031, 296.45263671875, 342.21826171875, 298.08880615234375,
         -341.65997314453125, 440.3846130371094, -343.41845703125, -299.17828369140625, -344.0526428222656]]

    bias1 = torch.FloatTensor(bias1)
    bias2 = torch.FloatTensor(bias2)
    bias3 = torch.FloatTensor(bias3)
    weight1 = torch.FloatTensor(weight1)
    weight2 = torch.FloatTensor(weight2)
    weight3 = torch.FloatTensor(weight3)

    controller = FFNNC()
    controller.layer1.weight = torch.nn.Parameter(weight1)
    controller.layer2.weight = torch.nn.Parameter(weight2)
    controller.layer3.weight = torch.nn.Parameter(weight3)
    controller.layer1.bias = torch.nn.Parameter(bias1)
    controller.layer2.bias = torch.nn.Parameter(bias2)
    controller.layer3.bias = torch.nn.Parameter(bias3)

    control_input_list = [[-0.1, -0.1, 7.81],
                          [-0.1, -0.1, 11.81],
                          [-0.1, 0.1, 7.81],
                          [-0.1, 0.1, 11.81],
                          [0.1, -0.1, 7.81],
                          [0.1, -0.1, 11.81],
                          [0.1, 0.1, 7.81],
                          [0.1, 0.1, 11.81]]
    init = initalCondition
    trajectory = [init]
    r = ode(func1)
    r.set_initial_value(init)
    ex_list = []
    ey_list = []
    ez_list = []
    t = 0
    time = [t]
    trace = [[t]]
    trace[0].extend(init[3:])
    i = 0
    while t <= time_bound:
        ex = trajectory[i][3] - trajectory[i][0]
        ey = trajectory[i][4] - trajectory[i][1]
        ez = trajectory[i][5] - trajectory[i][2]
        evx = trajectory[i][6] - ref_input[0]
        evy = trajectory[i][7] - ref_input[1]
        evz = trajectory[i][8] - ref_input[2]

        sc = ref_input[3] # math.atan2(dot, det)

        tmp1 = ex * math.cos(sc) - ey * math.sin(sc)
        tmp2 = ex * math.sin(sc) + ey * math.cos(sc)
        ex = tmp1
        ey = tmp2

        tmp1 = evx * math.cos(sc) - evy * math.sin(sc)
        tmp2 = evx * math.sin(sc) + evy * math.cos(sc)
        evx = tmp1
        evy = tmp2

        data = torch.FloatTensor([0.2 * ex, 0.2 * ey, 0.2 * ez, 0.1 * evx, 0.1 * evy, 0.1 * evz])
        res = controller(data)
        res = res.detach().numpy()
        idx = np.argmax(res)
        u = control_input_list[idx] + ref_input[0:3] + [sc]

        init = trajectory[i]
        r = ode(func1)
        r.set_initial_value(init)
        r.set_f_params(u)
        val = r.integrate(r.t + time_step)

        t += time_step
        i += 1
        #  print(i,idx,u,res)
        trajectory.append(val)
        time.append(t)

        ex_list.append(ex)
        ey_list.append(ey)
        ez_list.append(ez)
        trace.append([t])
        trace[i].extend(val[3:])  # remove the reference trajectory from the trace
    return trace


# mode, initial_state, time_step, time_bound, x_d
def TC_Simulate(Mode,initialCondition,time_bound):
    time_step = 0.05;
    time_bound = float(time_bound)
    Mode = Mode[1:-1]
    mode_parameters = Mode.split(";")
    mode_parameters = [float(x) for x in mode_parameters]
    number_points = int(np.ceil(time_bound / time_step))
    t = [i * time_step for i in range(0, number_points)]
    if t[-1] != time_step:
        t.append(time_bound)
    newt = []
    for step in t:
        newt.append(float(format(step, '.4f')))
    t = np.array(newt)

    ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
    ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
    ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_bound
    sym_rot_angle = 0
    trace = runModel(mode_parameters[0:3] + list(initialCondition), time_bound, time_step, [ref_vx, ref_vy, ref_vz,
                                                                                         sym_rot_angle])
    return np.array(trace)

# mode, initial_state, time_step, time_bound, x_d
def TC_Simulate_Batch(Mode, initialConditions,time_bound):
    return np.stack(tuple(map(lambda x: TC_Simulate(Mode, x, time_bound), initialConditions)))

def get_transform_information(waypoint: Waypoint) -> Tuple[np.array, float]:
    mode: str = waypoint.mode
    mode_parameters: Optional[List[float]] = waypoint.mode_parameters
    time_bound = waypoint.time_bound
    if mode != "follow_waypoint":
        raise NotImplementedError("haven't implemented modes other than follow waypoint for these dynamics")
    # old_center = prev_mode_parameters
    dot = (mode_parameters[3] - mode_parameters[0])
    det = (mode_parameters[4] - mode_parameters[1])
    # detz = (mode_parameters[5] - mode_parameters[2])
    dir_angle = math.atan2(dot, det)
    dir_angle_yz = 0 # math.atan2(det, detz)
    ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
    ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
    ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_bound
    translation_vector: np.array = np.zeros((6,))
    translation_vector[:3] = -1 * np.array(mode_parameters[3:])
    translation_vector[3:] = -1 * np.array([ref_vx, ref_vy, ref_vz])
    return translation_vector, dir_angle, dir_angle_yz

# def get_transform_information(mode_parameters: List[float], initset: pc.Polytope) -> Tuple[np.array, float]:
#    pass


def transform_poly_to_virtual(poly, transform_information):
    translation_vector, new_system_angle, new_system_angle_yz = transform_information
    poly_out: pc.Polytope = poly.translation(translation_vector)
    poly_out = poly_out.rotation(i=0, j=1, theta=new_system_angle)
    # poly_out.rotation(i=1, j=2, theta=new_system_angle_yz)
    return poly_out.rotation(i=3, j=4, theta=new_system_angle)
    #return poly_out.rotation(i=4, j=5, theta=new_system_angle_yz)


def transform_mode_to_virtual(waypoint: Waypoint, transform_information):
    point = waypoint.mode_parameters
    xs1 = point[0]  # x_i
    ys1 = point[1]  # y_i
    zs1 = point[2]
    xd1 = point[3]
    yd1 = point[4]
    zd1 = point[5]
    translation_vector, sc, sc_yz = transform_information
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    z_n = translation_vector[2]
    xs2 = (xs1 + x_n) * math.cos(sc) - (ys1 + y_n) * math.sin(sc)
    ys2 = (xs1 + x_n) * math.sin(sc) + (ys1 + y_n) * math.cos(sc)
    xd2 = (xd1 + x_n) * math.cos(sc) - (yd1 + y_n) * math.sin(sc)
    yd2 = (xd1 + x_n) * math.sin(sc) + (yd1 + y_n) * math.cos(sc)
    zs3 = zs1 + z_n
    zd3 = zd1 + z_n
    #ys3 = ys2 * math.cos(sc_yz) - (zs1 + z_n) * math.sin(sc_yz)
    #zs3 = ys2 * math.sin(sc_yz) + (zs1 + z_n) * math.sin(sc_yz)
    #yd3 = yd2 * math.cos(sc_yz) - (zd1 + z_n) * math.sin(sc_yz)
    #zd3 = yd2 * math.sin(sc_yz) + (zd1 + z_n) * math.sin(sc_yz)
    xs2 = round(xs2)
    ys3 = round(ys2) # round(ys3)
    zs3 = round(zs3) # round(zs3)
    xd2 = round(xd2)
    yd3 = round(yd2) # round(yd3)
    zd3 = round(zd3) # round(zd3)
    return Waypoint(waypoint.mode,[xs2, ys3, zs3, xd2, yd3, zd3], waypoint.time_bound, waypoint.id)


def transform_poly_from_virtual(poly, transform_information):
    new_system_angle = -1 * transform_information[1]
    translation_vector = -1 * transform_information[0]
    out_poly = poly.rotation(i=3, j=4, theta=new_system_angle)
    out_poly = out_poly.rotation(i=0, j=1, theta=new_system_angle)
    return out_poly.translation(translation_vector) # out_poly


def transform_mode_from_virtual(waypoint: Waypoint, transform_information):
    point = waypoint.mode_parameters
    sc = -1 * transform_information[1]
    translation_vector = -1 * transform_information[0]
    xs1 = point[0]  # x_i
    ys1 = point[1]  # y_i
    zs1 = point[2]
    xd1 = point[3]
    yd1 = point[4]
    zd1 = point[5]
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    z_n = translation_vector[2]
    xs2 = (xs1) * math.cos(sc) - (ys1) * math.sin(sc) + x_n
    ys2 = (xs1) * math.sin(sc) + (ys1) * math.cos(sc) + y_n
    xd2 = (xd1) * math.cos(sc) - (yd1) * math.sin(sc) + x_n
    yd2 = (xd1) * math.sin(sc) + (yd1) * math.cos(sc) + y_n
    zs3 = zs1 + z_n
    zd3 = zd1 + z_n
    #ys3 = (ys2) * math.cos(sc_yz) - (zs1) * math.sin(sc_yz)
    #zs3 = (ys2) * math.sin(sc_yz) + (zs1) * math.cos(sc_yz) + z_n
    #yd3 = (yd2) * math.cos(sc_yz) - (zd1) * math.sin(sc_yz)
    #zd3 = (yd2) * math.sin(sc_yz) + (zd1) * math.cos(sc_yz) + z_n
    xs2 = round(xs2)
    ys3 = round(ys2)  # round(ys3)
    zs3 = round(zs3)  # round(zs3)
    xd2 = round(xd2)
    yd3 = round(yd2)  # round(yd3)
    zd3 = round(zd3)  # round(zd3)
    return Waypoint(waypoint.mode,[xs2, ys3, zs3, xd2, yd3, zd3], waypoint.time_bound, waypoint.id)
    # x2 = x1 * math.cos(sc) + y1 * math.sin(sc) - x_n
    # y2 = -1 * x1 * math.sin(sc) + y1 * math.cos(sc) - y_n
    # x2 = x1 - x_n
    # y2 = y1 - y_n
    # return [x2, y2]


def transform_state_from_then_to_virtual_dryvr_string(point, transform_information_from, transform_information_to):
    pass


def get_virtual_mode_parameters():
    pass


def get_flowstar_parameters(mode_parameters: List[float], initial_set: np.array, time_step: float, time_bound: float,
                            mode: str):
    if mode != "follow_waypoint":
        raise NotImplementedError("These linear dynamics only support waypoint following mode")
    num_vars = 4
    order = 4
    hyper_params = [str(num_vars), str(time_step), str(time_bound), str(order)]
    cur_list: List[str] = hyper_params[:]
    var_names: List[str] = ["t", "x", "y", "z"]
    cur_list.extend(var_names[:num_vars])
    ode_rhs = ["1"]
    ode_rhs.extend([' + '.join([str(val) + " * (" + var_names[ind + 1] + ' - ' + "(" + str(mode_parameters[ind]) + "))"
                                for ind, val in enumerate(A_row)]) for A_row in A])
    cur_list.extend(ode_rhs)
    # time initset lowerbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[0, :]])
    # time initset upperbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[1, :]])
    return cur_list


def get_sherlock_parameters(mode_parameters: List[float], initial_set: np.array, time_step: float, time_bound: float,
                            mode: str):
    if mode != "follow_waypoint":
        raise NotImplementedError("These quadrotor dynamics only support waypoint following mode")
    num_nn_outputs = 1
    num_nn_inputs = initial_set.shape[1]
    num_vars = num_nn_inputs + num_nn_outputs
    order = 4
    nn_ctrl_file_path = "../systems_with_networks/Ex_Quadrotor/trial_controller_3"
    hyper_params = [str(num_vars), str(time_step), str(time_bound), str(order), nn_ctrl_file_path, num_nn_inputs,
                    num_nn_outputs]
    cur_list: List[str] = hyper_params[:]
    var_names: List[str] = ["t", "x", "y", "z"]
    cur_list.extend(var_names[:num_vars])
    ode_rhs = ["1"]
    ode_rhs.extend([' + '.join([str(val) + " * (" + var_names[ind + 1] + ' - ' + "(" + str(mode_parameters[ind]) + "))"
                                for ind, val in enumerate(A_row)]) for A_row in A])
    cur_list.extend(ode_rhs)
    # time initset lowerbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[0, :]])
    # time initset upperbound
    cur_list.append('0')
    cur_list.extend([str(val) for val in initial_set[1, :]])
    return cur_list

def get_flowstar_parameters(mode_parameters: List[float], initial_set: np.array, time_step: float, time_bound: float, mode: str):
    initial_condition = {'x1':[-0.2,-0.2],
                        'x2':[-0.2,-0.2],
                        'x3':[0,0],
                        'x4':[0,0],
                        'x5':[0,0],
                        'x6':[0,0],
                        'x7':[0.2,0.2],
                        'x8':[-0.2,-0.2],
                        'x9':[0,0]}

    ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_bound
    ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_bound
    ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_bound

    # Initial condition for verisig is ex,ey,ez,vx,vy,vz,x,y,z
    initial_condition['x1'][0] = initial_set[0, 0] - mode_parameters[0]
    initial_condition['x1'][1] = initial_set[1, 0] - mode_parameters[0]
    initial_condition['x2'][0] = initial_set[0, 1] - mode_parameters[1]
    initial_condition['x2'][1] = initial_set[1, 1] - mode_parameters[1]
    initial_condition['x3'][0] = initial_set[0, 2] - mode_parameters[2]
    initial_condition['x3'][1] = initial_set[1, 2] - mode_parameters[2]
    initial_condition['x4'][0] = initial_set[0, 3]
    initial_condition['x4'][1] = initial_set[1, 3]
    initial_condition['x5'][0] = initial_set[0, 4]
    initial_condition['x5'][1] = initial_set[1, 4]
    initial_condition['x6'][0] = initial_set[0, 5]
    initial_condition['x6'][1] = initial_set[1, 5]
    initial_condition['x7'][0] = initial_set[0, 0]
    initial_condition['x7'][1] = initial_set[1, 0]
    initial_condition['x8'][0] = initial_set[0, 1]
    initial_condition['x8'][1] = initial_set[1, 1]
    initial_condition['x9'][0] = initial_set[0, 2]
    initial_condition['x9'][1] = initial_set[1, 2]

    ref_vx = (mode_parameters[3]-mode_parameters[0])/time_bound
    ref_vy = (mode_parameters[4]-mode_parameters[1])/time_bound
    ref_vz = (mode_parameters[5]-mode_parameters[2])/time_bound
    dynamics_string = f"x1' == x4 - ({ref_vx}+0) & \nx2' == x5 - ({ref_vy}+0) & \nx3' == x6 - ({ref_vz}+0) & \nx4' == 9.81 * sin(u1) / cos(u1) & \nx5' == - 9.81 * sin(u2) / cos(u2) & \nx6' == u3 - 9.81 & \nx7' = x4 & \nx8' = x5 & \nx9' = x6 & \nclock' == 1 & \nt' == 1"

    return initial_condition, dynamics_string

def transform_trace_from_virtual(trace: np.array, transform_information):
    sc = -1*transform_information[1]
    translation_vector = -1*transform_information[0]
    x1 = trace[:,0:1]
    y1 = trace[:,1:2]
    z1 = trace[:,2:3]
    x_n = translation_vector[0]
    y_n = translation_vector[1]
    z_n = translation_vector[2]
    x2 = x1*math.cos(sc) - y1*math.sin(sc) + x_n
    y2 = x1*math.sin(sc) + y1*math.cos(sc) + y_n
    z3 = z1 + z_n
    res = np.concatenate((x2,y2,z3), axis=1)
    return res