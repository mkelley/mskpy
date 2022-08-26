from warnings import warn
import photutils

try:
    from ginga.GingaPlugin import LocalPlugin
    from ginga.gw import Widgets
except ImportError:
    warn('ginga is not present: ginga plugins will not run.',)

    class MSKpyAstrometry(LocalPlugin):
        pass
    Widgets = None

# https://ginga.readthedocs.io/en/stable/dev_manual/developers.html#writing-plugins-for-the-reference-viewer


class MSKpyAstrometry(LocalPlugin):

    def __init__(self, fv, fitsimage):
        super(MSKpyAstrometry, self).__init__(fv, fitsimage)

        self.dc = fv.get_draw_classes()
        canvas = self.dc.Canvas()
        canvas.name = 'Centroids'
        self.centroid_canvas = canvas

        canvas = self.dc.Canvas()
        canvas.name = 'Catalog'
        self.catalog_canvas = canvas

    def build_gui(self, container):
        top = Widgets.VBox()
        top.set_border_width(4)

        # this is a little trick for making plugins that work either in
        # a vertical or horizontal orientation.  It returns a box container,
        # a scroll widget and an orientation ('vertical', 'horizontal')
        vbox, sw, orientation = Widgets.get_oriented_box(container)
        vbox.set_border_width(4)
        vbox.set_spacing(2)

        # Show some instructions
        self.msgFont = self.fv.getFont("sansFont", 12)
        tw = Widgets.TextArea(wrap=True, editable=False)
        tw.set_font(self.msgFont)
        self.tw = tw

        # Frame for instructions and add the text widget with another
        # blank widget to stretch as needed to fill emp
        fr = Widgets.Frame("Instructions")
        vbox2 = Widgets.VBox()
        vbox2.add_widget(tw)
        vbox2.add_widget(Widgets.Label(""), stretch=1)
        fr.set_widget(vbox2)
        vbox.add_widget(fr, stretch=0)

        # target center panel
        frame = Widgets.Frame('Target center')
        hbox = Widgets.HBox()
        w, b = Widgets.build_info((('Select target', 'button'),
                                   ('X guess:', 'label', 'X guess', 'entry'),
                                   ('Y guess:', 'label', 'Y guess', 'entry'),
                                   ))
        b.select_target.add_callback('activated', self.select_target_callback)
        hbox.add_widget(w)

        w, b = Widgets.build_info(
            (('Centroid box:', 'label', 'centroid box', 'entry', 'Centroid', 'button'),
             ('X (pix):', 'label', 'Y (pix):', 'label'),
             ('RA (hr):', 'label', 'Dec (deg):', 'label'),
             ))
        b.centroid.add_callback('activated', self.centroid_callback)
        b.centroid_box.set_text('7')
        self.w.update(b)
        hbox.add_widget(w)

        frame.set_widget(hbox)
        vbox.add_widget(frame)

        # Add a spacer to stretch the rest of the way to the end of the
        # plugin space
        spacer = Widgets.Label('')
        vbox.add_widget(spacer, stretch=1)

        # scroll bars will allow lots of content to be accessed
        top.add_widget(sw, stretch=1)

        # A button box that is always visible at the bottom
        btns = Widgets.HBox()
        btns.set_spacing(3)

        # Add a close button for the convenience of the user
        btn = Widgets.Button("Close")
        btn.add_callback('activated', lambda w: self.close())
        btns.add_widget(btn, stretch=0)
        btns.add_widget(Widgets.Label(''), stretch=1)
        top.add_widget(btns, stretch=0)

        # Add our GUI to the container
        container.add_widget(top, stretch=1)

    def select_target_callback(self, w):
        pass

    def centroid_callback(self, w):
        im = self.fitsimage.get_image().get_data()

        try:
            xc = int(float(self.w.x_center.get_text()))
            yc = int(float(self.w.y_center.get_text()))
            box = int(float(self.w.centroid_box.get_text()))
        except ValueError:
            return

        if box < 3:
            box = 3
            self.w.centroid_box.set_text('3')
        elif box > min(xc, yc, im.shape[0] - yc, im.shape[1] - xc):
            box = int(min(xc, yc, im.shape[0] - yc, im.shape[1] - xc))
            self.w.centroid_box.set_text(str(box))

        x0 = xc - box // 2
        y0 = yc - box // 2

        subim = im[y0:y0 + box + 1, x0:x0 + box + 1]
        cxy = photutils.centroid_2dg(subim)
        self.w.x_center.set_text('{:.2f}'.format(cxy[0] + x0))
        self.w.y_center.set_text('{:.2f}'.format(cxy[1] + y0))

    def close(self):
        """
        Example close method.  You can use this method and attach it as a
        callback to a button that you place in your GUI to close the plugin
        as a convenience to the user.
        """
        chname = self.fv.get_channel_name(self.fitsimage)
        self.fv.stop_local_plugin(chname, str(self))
        return True

    def start(self):
        """
        This method is called just after ``build_gui()`` when the plugin
        is invoked.  This method may be called many times as the plugin is
        opened and closed for modal operations.  This method may be omitted
        in many cases.
        """
        self.tw.set_text("""Astrometry""")
        self.resume()

    def pause(self):
        """
        This method is called when the plugin loses focus.
        It should take any actions necessary to stop handling user
        interaction events that were initiated in ``start()`` or
        ``resume()``.
        This method may be called many times as the plugin is focused
        or defocused.  It may be omitted if there is no user event handling
        to disable.
        """
        pass

    def resume(self):
        """
        This method is called when the plugin gets focus.
        It should take any actions necessary to start handling user
        interaction events for the operations that it does.
        This method may be called many times as the plugin is focused or
        defocused.  The method may be omitted if there is no user event
        handling to enable.
        """
        pass

    def stop(self):
        """
        This method is called when the plugin is stopped.
        It should perform any special clean up necessary to terminate
        the operation.  The GUI will be destroyed by the plugin manager
        so there is no need for the stop method to do that.
        This method may be called many  times as the plugin is opened and
        closed for modal operations, and may be omitted if there is no
        special cleanup required when stopping.
        """
        pass

    def redo(self):
        """
        This method is called when the plugin is active and a new
        image is loaded into the associated channel.  It can optionally
        redo the current operation on the new image.  This method may be
        called many times as new images are loaded while the plugin is
        active.  This method may be omitted.
        """
        pass

    def __str__(self):
        return 'MSKpyAstrometry'
