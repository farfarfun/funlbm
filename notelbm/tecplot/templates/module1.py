import tecplot as tp
from tecplot.constant import *
from notelbm.tecplot.utils.connect import new_layout_connect

new_layout_connect()

# Uncomment the following line to connect to a running instance of Tecplot 360:
# tp.session.connect()

tp.macro.execute_command("""$!ReadDataSet  '\"/Users/chen/workspace/临时结果/V350000.dat\" '
  ReadDataOption = New
  ResetStyle = No
  VarLoadMode = ByName
  AssignStrandIDs = Yes
  VarNameList = '\"V1\" \"V2\" \"V3\" \"V4\" \"V5\" \"V6\" \"V7\"'""")
tp.macro.execute_command("""$!ReadDataSet  '\"/Users/chen/workspace/临时结果/Particles_350000.dat\" '
  ReadDataOption = Append
  ResetStyle = No
  VarLoadMode = ByName
  AssignStrandIDs = Yes
  VarNameList = '\"V1\";\"X\" \"V2\";\"Y\" \"V3\" \"V4\" \"V5\" \"V6\" \"V7\" \"T\"'""")
tp.active_frame().plot().rgb_coloring.red_variable_index = 2
tp.active_frame().plot().rgb_coloring.green_variable_index = 2
tp.active_frame().plot().rgb_coloring.blue_variable_index = 2
tp.active_frame().plot().contour(0).variable_index = 2
tp.active_frame().plot().contour(1).variable_index = 3
tp.active_frame().plot().contour(2).variable_index = 4
tp.active_frame().plot().contour(3).variable_index = 5
tp.active_frame().plot().contour(4).variable_index = 6
tp.active_frame().plot().contour(5).variable_index = 7
tp.active_frame().plot().contour(6).variable_index = 2
tp.active_frame().plot().contour(7).variable_index = 2
tp.active_frame().plot().show_contour = True
tp.active_frame().plot().contour(0).variable_index = 3
tp.active_frame().plot().contour(0).levels.reset_levels(
    [-1e-05, -8e-06, -6e-06, -4e-06, -2e-06, 0, 2e-06, 4e-06, 6e-06, 8e-06, 1e-05, 1.2e-05, 1.4e-05, 1.6e-05, 1.8e-05,
     2e-05])
tp.active_frame().plot().contour(1).variable_index = 7
tp.active_frame().plot().contour(1).levels.reset_levels([0])
tp.active_frame().plot().contour(1).colormap_name = 'GrayScale'
tp.macro.execute_command('''$!Pick AddAtPosition
  X = 5.53987005316
  Y = 5.25738334318
  ConsiderStyle = Yes''')
tp.macro.execute_command('$!Pick Copy')
tp.macro.execute_command('$!Pick Clear')
tp.macro.execute_command('''$!AttachGeom 
  AnchorPos
    {
    X = 400.694497
    Y = 250
    }
  Color = Custom30
  FillColor = Black
  LineThickness = 0.4
  ArrowheadAttachment = AtEnd
  ArrowheadSize = 2
  RawData
1
2
0 0 
14.1102828979 0''')
tp.active_frame().plot().view.zoom(xmin=216.255,
                                   xmax=615.295,
                                   ymin=78.7593,
                                   ymax=442.92)
tp.active_frame().plot().view.zoom(xmin=292.456,
                                   xmax=491.976,
                                   ymin=163.522,
                                   ymax=345.602)
tp.macro.execute_command('''$!Pick SetMouseMode
  MouseMode = Select''')
tp.macro.execute_command('''$!Pick AddAtPosition
  X = 5.79503839338
  Y = 4.60883047844
  ConsiderStyle = Yes''')
tp.active_frame().plot().fieldmaps(1).contour.flood_contour_group_index = 1
tp.active_frame().plot().axes.x_axis.min = 300.313
tp.active_frame().plot().axes.x_axis.max = 499.833
tp.active_frame().plot().axes.y_axis.min = 161.322
tp.active_frame().plot().axes.y_axis.max = 343.402
tp.active_frame().plot(PlotType.Cartesian2D).vector.u_variable_index = 3
tp.active_frame().plot(PlotType.Cartesian2D).vector.v_variable_index = 4
tp.active_frame().plot().streamtraces.timing.reset_delta()
tp.active_frame().plot().show_streamtraces = True
tp.active_frame().plot().streamtraces.add(seed_point=[418.777, 222.196],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.color = Color.White
tp.active_frame().plot().streamtraces.add(seed_point=[385.465, 201.455],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[412.492, 178.828],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[402.121, 290.078],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[409.978, 321.505],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[420.034, 285.05],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[445.804, 208.054],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[427.577, 274.05],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[411.235, 236.024],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[409.349, 262.423],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[364.723, 215.911],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[349.324, 275.936],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[321.355, 180.085],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[318.841, 315.219],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[399.921, 333.761],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[402.121, 165.943],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().streamtraces.add(seed_point=[470.631, 251.423],
                                          stream_type=Streamtrace.TwoDLine)
tp.active_frame().plot().contour(0).levels.add([-3.64811e-07])
tp.active_frame().plot().contour(0).levels.add([-2.38546e-07])
tp.active_frame().plot().contour(0).levels.add([4.8665e-07])
tp.active_frame().plot().contour(0).levels.add([2.66247e-07])
tp.macro.execute_command('''$!Pick SetMouseMode
  MouseMode = Select''')
tp.macro.execute_command('''$!Pick AddAtPosition
  X = 2.16952155936
  Y = 4.74704666273
  ConsiderStyle = Yes''')
tp.active_frame().plot().axes.y_axis.show = False
tp.active_frame().plot().axes.x_axis.show = False
tp.macro.execute_command('''$!Pick AddAtPosition
  X = 9.3780271707
  Y = 2.00398700532
  ConsiderStyle = Yes''')
tp.macro.execute_command('$!Pick Copy')
tp.active_frame().plot().contour(1).legend.show = False
tp.active_frame().plot().contour(0).legend.show = False
tp.macro.execute_command('''$!Pick AddAtPosition
  X = 9.50561134082
  Y = 8.23434731246
  ConsiderStyle = Yes''')
tp.macro.execute_command('''$!FrameControl ActivateByNumber
  Frame = 1''')
tp.active_frame().plot().frame.transparent = True
tp.active_frame().plot().frame.show_border = False
tp.active_frame().plot().axes.y_axis.min = 161.636
tp.active_frame().plot().axes.y_axis.max = 343.717
# End Macro.
