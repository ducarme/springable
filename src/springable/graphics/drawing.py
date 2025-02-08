from ..mechanics.node import Node
from ..mechanics.element import Element
from ..mechanics.mechanical_behavior import BivariateBehavior
from ..mechanics.assembly import Assembly
from ..mechanics.model import Model
from ..mechanics import shape
from .default_graphics_settings import AssemblyAppearanceOptions
from .custom_markers import HORIZONTAL_CART, VERTICAL_CART, ANCHOR
from .visual_helpers import compute_zigzag_line, compute_arc_line, compute_pathpatch_vertices, compute_coil_line, \
    compute_coil_arc
from scipy.interpolate import interp1d
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches
import numpy as np


class Drawing:
    def __init__(self, ax: plt.Axes, assembly_appearance: AssemblyAppearanceOptions):
        self._ax = ax
        self._aa = assembly_appearance

    def update(self, *args):
        raise NotImplementedError


class NodeDrawing(Drawing):

    def __init__(self, ax: plt.Axes, _node: Node, assembly_appearance: AssemblyAppearanceOptions):
        super().__init__(ax, assembly_appearance)
        self._node = _node

        if self._aa.node_style == 'elegant':
            marker = 'o'
            if self._node.is_fixed_horizontally() and self._node.is_fixed_vertically():
                triangle_marker = ANCHOR
            elif self._node.is_fixed_horizontally():
                triangle_marker = VERTICAL_CART
            elif self._node.is_fixed_vertically():
                triangle_marker = HORIZONTAL_CART
            else:
                triangle_marker = None
        else:
            triangle_marker = None
            if self._node.is_fixed_horizontally() and self._node.is_fixed_vertically():
                marker = 's'
            elif self._node.is_fixed_horizontally():
                marker = '<'
            elif self._node.is_fixed_vertically():
                marker = '^'
            else:
                marker = 'o'

        self._text = None
        if self._aa.show_node_numbers:
            self._text = self._ax.text(self._node.get_x(), self._node.get_y(), f'{self._node.get_node_nb()}',
                                       color=self._aa.node_nb_color, weight='bold', zorder=5.5)

        self._triangle_graphic = None
        if triangle_marker is not None:  # do not use node_style for this conditional
            self._triangle_graphic = self._ax.plot([self._node.get_x()], [self._node.get_y()], zorder=4.0,
                                                   marker=triangle_marker,
                                                   markersize=6 * self._aa.node_size,
                                                   markerfacecolor='#cecece',
                                                   markeredgecolor=self._aa.node_color)[0]

        self._node_graphic = self._ax.plot([self._node.get_x()], [self._node.get_y()], zorder=5.0,
                                           marker=marker,
                                           markersize=self._aa.node_size,
                                           color=self._aa.node_color)[0]

    def update(self, *args):
        self._node_graphic.set_xdata([self._node.get_x()])
        self._node_graphic.set_ydata([self._node.get_y()])
        if self._aa.show_node_numbers:
            self._text.set_x(self._node.get_x())
            self._text.set_y(self._node.get_y())
        if self._triangle_graphic is not None:
            self._triangle_graphic.set_xdata([self._node.get_x()])
            self._triangle_graphic.set_ydata([self._node.get_y()])



class ShapeDrawing(Drawing):
    def __init__(self, _shape: shape.Shape, size: float | None, is_hysteron,
                 ax: plt.Axes, color: str, opacity: float, aa: AssemblyAppearanceOptions):
        super().__init__(ax, aa)
        self._shape = _shape
        self._size = size
        self._is_hysteron = is_hysteron
        self._color = color
        self._opacity = opacity
        self._hysteron_label_position = None  # to be set correctly in subclasses

    def update(self, color, opacity):
        if color is not None:
            self._color = color
        if opacity is not None:
            self._opacity = opacity
        # to be extended in subclasses

    def get_hysteron_label_position(self):
        return self._hysteron_label_position


# Here in below, core drawing classes for basic shapes

class SegmentDrawing(ShapeDrawing):

    def __init__(self, segment: shape.Segment, width: float, is_hysteron, ax: plt.Axes, color: str, opacity: float,
                 aa: AssemblyAppearanceOptions):
        super().__init__(segment, width, is_hysteron, ax, color, opacity, aa)
        x0, y0, x1, y1 = self._shape.get_nodal_coordinates()

        if self._aa.spring_style == 'elegant':
            x_coords, y_coords = compute_coil_line((x0, y0), (x1, y1), self._aa.spring_nb_coils,
                                                   self._size * self._aa.spring_width_scaling,
                                                   aspect=self._aa.spring_aspect)
        elif self._aa.spring_style == 'line':
            x_coords, y_coords = [x0, x1], [y0, y1]
        else:  # "basic" or anything else
            x_coords, y_coords = compute_zigzag_line((x0, y0), (x1, y1), 8 + 2 * self._aa.spring_nb_coils,
                                                     self._size * self._aa.spring_width_scaling)

        self._graphics = self._ax.plot(x_coords, y_coords, lw=self._aa.spring_linewidth,
                                       color=self._color, alpha=self._opacity, zorder=0.1)[0]
        self._hysteron_label_position = ((x0 + x1) / 2, (y0 + y1) / 2) if self._is_hysteron else None

    def update(self, color: str, opacity: float):
        super().update(color, opacity)
        x0, y0, x1, y1 = self._shape.get_nodal_coordinates()
        if self._aa.spring_style == 'elegant':
            x_coords, y_coords = compute_coil_line((x0, y0), (x1, y1), self._aa.spring_nb_coils,
                                                   self._size * self._aa.spring_width_scaling)
        elif self._aa.spring_style == 'line':
            x_coords, y_coords = [x0, x1], [y0, y1]
        else:  # "basic" or anything else
            x_coords, y_coords = compute_zigzag_line((x0, y0), (x1, y1), 8 + 2 * self._aa.spring_nb_coils,
                                                     self._size * self._aa.spring_width_scaling)
        self._graphics.set_xdata(x_coords)
        self._graphics.set_ydata(y_coords)
        if color is not None:
            self._graphics.set_color(color)
        if opacity is not None:
            self._graphics.set_alpha(opacity)
        if self._is_hysteron:
            self._hysteron_label_position = ((x0 + x1) / 2, (y0 + y1) / 2)


class AngleDrawing(ShapeDrawing):

    def __init__(self, _angle: shape.Angle, radius, is_hysteron, ax: plt.Axes,
                 color: str, opacity: float, aa: AssemblyAppearanceOptions):
        super().__init__(_angle, radius, is_hysteron, ax, color, opacity, aa)
        x0, y0, x1, y1, x2, y2 = self._shape.get_nodal_coordinates()
        center = (x1, y1)
        angle = shape.Angle.calculate_angle(x0, y0, x1, y1, x2, y2)
        end_angle = shape.Angle.calculate_angle(x1 + 1, y1, x1, y1, x2, y2)
        start_angle = end_angle - angle

        if self._aa.rotation_spring_style == 'elegant':
            x_coords, y_coords = compute_coil_arc(center, 0.8 * self._size * self._aa.rotation_spring_radius_scaling,
                                                  start_angle, end_angle,
                                                  self._aa.rotation_spring_nb_coils,
                                                  self._aa.rotation_spring_radii_ratio,
                                                  self._aa.rotation_spring_aspect)
        else:  # "line" or anything else
            x_coords, y_coords = compute_arc_line(center, self._size * self._aa.rotation_spring_radius_scaling,
                                                  start_angle, end_angle)

        self._graphics = self._ax.plot(x_coords, y_coords, lw=self._aa.rotation_spring_linewidth,
                                       color=self._color, alpha=self._opacity, zorder=0)[0]
        if self._is_hysteron:
            mid_angle = (start_angle + end_angle) / 2
            self._hysteron_label_position = (
                center[0] + self._size * self._aa.rotation_spring_radius_scaling * np.cos(mid_angle),
                center[0] + self._size * self._aa.rotation_spring_radius_scaling * np.sin(mid_angle)
            )
        else:
            self._hysteron_label_position = None

    def update(self, color: str, opacity: float):
        super().update(color, opacity)
        x0, y0, x1, y1, x2, y2 = self._shape.get_nodal_coordinates()
        center = (x1, y1)
        angle = shape.Angle.calculate_angle(x0, y0, x1, y1, x2, y2)
        end_angle = shape.Angle.calculate_angle(x1 + 1, y1, x1, y1, x2, y2)
        start_angle = end_angle - angle

        if self._aa.rotation_spring_style == 'elegant':
            x_coords, y_coords = compute_coil_arc(center, self._size * self._aa.rotation_spring_radius_scaling,
                                                  start_angle, end_angle,
                                                  self._aa.rotation_spring_nb_coils,
                                                  self._aa.rotation_spring_radii_ratio,
                                                  self._aa.rotation_spring_aspect)
        else:  # "line" or anything else
            x_coords, y_coords = compute_arc_line(center, self._size * self._aa.rotation_spring_radius_scaling,
                                                  start_angle, end_angle)
        self._graphics.set_xdata(x_coords)
        self._graphics.set_ydata(y_coords)
        if color is not None:
            self._graphics.set_color(color)
        if opacity is not None:
            self._graphics.set_alpha(opacity)

        if self._is_hysteron:
            mid_angle = (start_angle + end_angle) / 2
            self._hysteron_label_position = (
                center[0] + self._size * self._aa.rotation_spring_radius_scaling * np.cos(mid_angle),
                center[0] + self._size * self._aa.rotation_spring_radius_scaling * np.sin(mid_angle)
            )


class AreaDrawing(ShapeDrawing):
    def __init__(self, area: shape.Area, is_hysteron, ax: plt.Axes,
                 color: str, opacity: float, aa: AssemblyAppearanceOptions):
        super().__init__(area, None, is_hysteron, ax, color, opacity, aa)
        coordinates = self._shape.get_nodal_coordinates()
        x, y = coordinates[::2], coordinates[1::2]
        zorder = np.min(x) / np.abs(np.max(x) - np.min(x))
        self._graphics = self._ax.fill(x, y, color=self._color, alpha=self._opacity, zorder=zorder)[0]
        self._hysteron_label_position = (np.mean(x), np.mean(y)) if self._is_hysteron else None

    def update(self, color: str, opacity: float):
        super().update(color, opacity)
        coordinates = self._shape.get_nodal_coordinates()
        self._graphics.set_xy(np.array(coordinates).reshape(-1, 2))
        if color is not None:
            self._graphics.set_color(color)
        if opacity is not None:
            self._graphics.set_alpha(opacity)
        if self._is_hysteron:
            self._hysteron_label_position = (np.mean(coordinates[::2]), np.mean(coordinates[1::2]))


class PathDrawing(ShapeDrawing):
    def __init__(self, path: shape.Path, is_hysteron, ax: plt.Axes,
                 color: str, opacity: float, aa: AssemblyAppearanceOptions):
        super().__init__(path, None, is_hysteron, ax, color, opacity, aa)
        coordinates = self._shape.get_nodal_coordinates()
        x, y = coordinates[::2], coordinates[1::2]
        lengths = [0.0]
        for i in range(len(x) - 1):
            lengths.append(lengths[-1] + shape.Segment.calculate_length(x[i], y[i], x[i + 1], y[i + 1]))
        t = np.linspace(0, 1, 10) * lengths[-1]
        xx = interp1d(lengths, x)(t[1:-1])
        yy = interp1d(lengths, y)(t[1:-1])
        graphic0 = self._ax.plot(x, y, lw=self._aa.line_spring_linewidth,
                                 color=self._color, alpha=self._opacity, zorder=0)[0]
        graphic1 = self._ax.plot(xx, yy, ls='', markersize=self._aa.line_spring_linewidth * 0.4,
                                 marker='o', color='#CECECE', zorder=0.1)[0]
        self._graphics = (graphic0, graphic1)
        self._hysteron_label_position = (np.mean(x), np.mean(y)) if self._is_hysteron else None

    def update(self, color, opacity):
        super().update(color, opacity)
        graphic0, graphic1 = self._graphics
        coordinates = self._shape.get_nodal_coordinates()
        x, y = coordinates[::2], coordinates[1::2]
        lengths = [0.0]
        for i in range(len(x) - 1):
            lengths.append(lengths[-1] + shape.Segment.calculate_length(x[i], y[i], x[i + 1], y[i + 1]))
        t = np.linspace(0, 1, 10) * lengths[-1]
        xx = interp1d(lengths, x)(t[1:-1])
        yy = interp1d(lengths, y)(t[1:-1])
        graphic0.set_xdata(x)
        graphic0.set_ydata(y)
        graphic1.set_xdata(xx)
        graphic1.set_ydata(yy)
        if color is not None:
            graphic0.set_color(color)
        if opacity is not None:
            graphic0.set_alpha(opacity)
        if self._is_hysteron:
            self._hysteron_label_position = (np.mean(coordinates[::2]), np.mean(coordinates[1::2]))


class DistanceDrawing(ShapeDrawing):
    def __init__(self,
                 dist: shape.DistancePointLine | shape.SquaredDistancePointSegment | shape.SignedDistancePointLine,
                 is_hysteron, ax: plt.Axes, color: str, opacity: float, aa: AssemblyAppearanceOptions):
        super().__init__(dist, None, is_hysteron, ax, color, opacity, aa)
        x0, y0, x1, y1, x2, y2 = self._shape.get_nodal_coordinates()
        graphic0 = self._ax.plot(x0, y0, zorder=2.1, marker='.', markersize=self._aa.node_size / 2.0,
                                 alpha=self._opacity, color=self._color)[0]
        graphic1 = self._ax.plot([x1, x2], [y1, y2], lw=self._aa.distance_spring_line_linewidth,
                                 alpha=self._opacity, color=self._color, zorder=0)[0]
        graphic2 = self._ax.plot([x1, x2], [y1, y2], lw=0.75,
                                 color=self._aa.distance_spring_line_default_color, zorder=0)[0]
        self._graphics = (graphic0, graphic1, graphic2)
        self._hysteron_label_position = ((x1 + x2) / 2, (y1 + y2) / 2) if self._is_hysteron else None

    def update(self, color, opacity):
        super().update(color, opacity)
        graphic0, graphic1, graphic2 = self._graphics
        x0, y0, x1, y1, x2, y2 = self._shape.get_nodal_coordinates()
        graphic0.set_xdata([x0])
        graphic0.set_ydata([y0])
        graphic1.set_xdata([x1, x2])
        graphic1.set_ydata([y1, y2])
        graphic2.set_xdata([x1, x2])
        graphic2.set_ydata([y1, y2])
        if color is not None:
            graphic0.set_color(color)
            graphic1.set_color(color)
        if opacity is not None:
            graphic0.set_alpha(opacity)
            graphic1.set_alpha(opacity)
        if self._is_hysteron:
            self._hysteron_label_position = ((x1 + x2) / 2, (y1 + y2) / 2)


class HoleyAreaDrawing(ShapeDrawing):
    def __init__(self, holey_area: shape.HoleyArea,
                 is_hysteron, ax: plt.Axes, color: str, opacity: float, aa: AssemblyAppearanceOptions):
        super().__init__(holey_area, None, is_hysteron, ax, color, opacity, aa)
        self._bulk = holey_area.get_bulk_area()
        self._holes = holey_area.get_holes()

        bulk_coords = np.array([[node.get_x(), node.get_y()] for node in self._bulk.get_nodes()]).T
        holes_coords = [np.array([[node.get_x(), node.get_y()] for node in hole.get_nodes()]).T
                        for hole in self._holes]
        polys = [bulk_coords] + holes_coords
        vertices, codes = compute_pathpatch_vertices(polys)
        self._patch = matplotlib.patches.PathPatch(matplotlib.path.Path(vertices, codes),
                                                   fill=True, color=color, alpha=opacity)
        self._ax.add_patch(self._patch)
        self._hysteron_label_position = ((np.mean(bulk_coords[0, :]), np.mean(bulk_coords[1, :]))
                                         if self._is_hysteron else None)

    def update(self, color, opacity):
        super().update(color, opacity)
        bulk_coords = np.array([[node.get_x(), node.get_y()] for node in self._bulk.get_nodes()]).T
        holes_coords = [np.array([[node.get_x(), node.get_y()] for node in hole.get_nodes()]).T
                        for hole in self._holes]
        polys = [bulk_coords] + holes_coords
        vertices, _ = compute_pathpatch_vertices(polys, compute_code=False)
        self._patch.get_path().vertices[:] = vertices

        if color is not None:
            self._patch.set_color(color)
        if opacity is not None:
            self._patch.set_alpha(opacity)
        if self._is_hysteron:
            self._hysteron_label_position = (np.mean(bulk_coords[0, :]), np.mean(bulk_coords[1, :]))


class CompoundDrawing(ShapeDrawing):
    """ Class to draw shapes that are (eventually) made of sub-shapes that can be drawn
    (that is, belonging to the dictionary DRAWABLE_SHAPES """

    DRAWABLE_SHAPES = {shape.Segment: SegmentDrawing,
                       shape.Angle: AngleDrawing,
                       shape.Area: AreaDrawing,
                       shape.HoleyArea: HoleyAreaDrawing,
                       shape.Path: PathDrawing,
                       shape.DistancePointLine: DistanceDrawing,
                       shape.SignedDistancePointLine: DistanceDrawing,
                       shape.SquaredDistancePointSegment: DistanceDrawing
                       }

    def __init__(self, cs: shape.CompoundShape,
                 is_hysteron, ax: plt.Axes, color: str, opacity: float, aa: AssemblyAppearanceOptions):
        super().__init__(cs, None, is_hysteron, ax, color, opacity, aa)
        self._drawings = []
        for subshape in cs.get_shapes():
            if type(subshape) in CompoundDrawing.DRAWABLE_SHAPES:
                drawing_type = CompoundDrawing.DRAWABLE_SHAPES[type(subshape)]
                drawing = drawing_type(subshape, False, self._ax, self._color, self._opacity, self._aa)
            else:
                drawing = CompoundDrawing(subshape, False, self._ax, self._color, self._opacity, self._aa)
            self._drawings.append(drawing)

        if self._is_hysteron:
            coord = self._shape.get_nodal_coordinates()
            self._hysteron_label_position = np.mean(coord[::2]), np.mean(coord[1::2])
        else:
            self._hysteron_label_position = None

    def update(self, color, opacity):
        super().update(color, opacity)
        for drawing in self._drawings:
            drawing.update(color, opacity)
        if self._is_hysteron:
            coord = self._shape.get_nodal_coordinates()
            self._hysteron_label_position = np.mean(coord[::2]), np.mean(coord[1::2])


class ElementDrawing(Drawing):

    def __init__(self, ax: plt.Axes, _element: Element, width, assembly_appearance: AssemblyAppearanceOptions,
                 color_handler=None, opacity_handler=None):
        super().__init__(ax, assembly_appearance)
        self._element = _element
        self._color_handler = color_handler
        self._opacity_handler = opacity_handler
        self._width = width
        behavior = self._element.get_behavior()
        self._hysteron_info = behavior.get_hysteron_info() if isinstance(behavior, BivariateBehavior) else {}
        self._is_hysteron = True if self._hysteron_info else False

        # CREATE GRAPHICS FOR DRAWING
        self._shape_drawing, self._hysteron_state_bg_graphic, self._hysteron_state_id_graphic = self._make()

    def _make(self):
        if self._color_handler is None:
            color = None  # to specify later
        else:
            color = self._color_handler.determine_property_value(self._element)

        if self._opacity_handler is None or not isinstance(self._element.get_shape(),
                                                           (
                                                                   shape.DistancePointLine,
                                                                   shape.SquaredDistancePointSegment,
                                                                   shape.SignedDistancePointLine)
                                                           ):
            opacity = None  # to specify later
        else:
            opacity = self._opacity_handler.determine_property_value(self._element)

        hysteron_state_drawing_position = None
        _shape = self._element.get_shape()
        if isinstance(_shape, shape.Segment):
            color = color if color is not None else self._aa.spring_default_color
            opacity = opacity if opacity is not None else self._aa.spring_default_opacity
            shape_drawing = SegmentDrawing(_shape, self._width, self._is_hysteron,
                                           self._ax, color, opacity, self._aa)

        elif isinstance(_shape, shape.Angle):
            color = color if color is not None else self._aa.rotation_spring_default_color
            opacity = opacity if opacity is not None else self._aa.rotation_spring_default_opacity
            shape_drawing = AngleDrawing(_shape, self._width, self._is_hysteron,
                                         self._ax, color, opacity, self._aa)
        elif isinstance(_shape, shape.Area):
            color = color if color is not None else self._aa.area_spring_default_color
            opacity = opacity if opacity is not None else self._aa.area_spring_default_opacity
            shape_drawing = AreaDrawing(_shape, self._is_hysteron,
                                        self._ax, color, opacity, self._aa)

        elif isinstance(_shape, shape.Path):
            color = color if color is not None else self._aa.line_spring_default_color
            opacity = opacity if opacity is not None else self._aa.line_spring_default_opacity
            shape_drawing = PathDrawing(_shape, self._is_hysteron,
                                        self._ax, color, opacity, self._aa)
        elif isinstance(_shape,
                        (shape.DistancePointLine, shape.SquaredDistancePointSegment, shape.SignedDistancePointLine)):
            color = color if color is not None else self._aa.distance_spring_line_default_color
            opacity = opacity if opacity is not None else self._aa.distance_spring_line_default_opacity
            shape_drawing = DistanceDrawing(_shape, self._is_hysteron,
                                            self._ax, color, opacity, self._aa)
        elif isinstance(_shape, shape.HoleyArea):
            color = color if color is not None else self._aa.area_spring_default_color
            opacity = opacity if opacity is not None else self._aa.area_spring_default_opacity
            shape_drawing = HoleyAreaDrawing(_shape, self._is_hysteron, self._ax, color, opacity, self._aa)
        else:
            raise NotImplementedError(
                f'Cannot draw element because no implementation of how to draw its shape {_shape}')

        hysteron_state_drawing_position = shape_drawing.get_hysteron_label_position()

        if self._aa.show_state_of_hysterons and self._hysteron_info:
            internal_coord = self._element.get_internal_coordinates()
            for i, interval in enumerate(self._hysteron_info['branch_intervals']):
                if interval[0] <= internal_coord <= interval[1]:
                    state_id = self._hysteron_info['branch_ids'][i]
                    break
            else:
                raise ValueError('Cannot determine state of hysteron')
            hysteron_state_bg_graphic = self._ax.plot(*hysteron_state_drawing_position, 'o', zorder=1.0,
                                                      markersize=self._aa.hysteron_state_label_size,
                                                      color=self._aa.hysteron_state_bg_color, markeredgecolor=color,
                                                      markeredgewidth=self._aa.spring_linewidth)[0]
            hysteron_state_id_graphic = self._ax.annotate(state_id, xy=hysteron_state_drawing_position,
                                                          color=self._aa.hysteron_state_txt_color,
                                                          fontsize=0.65 / min(2,
                                                                              len(state_id)) * self._aa.hysteron_state_label_size,
                                                          weight='bold',
                                                          verticalalignment="center",
                                                          horizontalalignment="center",
                                                          zorder=1.5)
        else:
            hysteron_state_bg_graphic = None
            hysteron_state_id_graphic = None
        return shape_drawing, hysteron_state_bg_graphic, hysteron_state_id_graphic

    def update(self):
        color = None
        opacity = None
        if self._color_handler is not None:
            color = self._color_handler.determine_property_value(self._element)
        if (self._opacity_handler is not None
                and isinstance(self._element.get_shape(),
                               (shape.DistancePointLine, shape.SquaredDistancePointSegment,
                                shape.SignedDistancePointLine))):
            opacity = self._opacity_handler.determine_property_value(self._element)

        self._shape_drawing.update(color, opacity)

        if self._hysteron_state_bg_graphic is not None and self._hysteron_state_id_graphic is not None:
            hysteron_state_graphic_position = self._shape_drawing.get_hysteron_label_position()
            self._hysteron_state_bg_graphic.set_xdata([hysteron_state_graphic_position[0]])
            self._hysteron_state_bg_graphic.set_ydata([hysteron_state_graphic_position[1]])
            internal_coord = self._element.get_internal_coordinates()
            for i, interval in enumerate(self._hysteron_info['branch_intervals']):
                if interval[0] <= internal_coord <= interval[1]:
                    state_id = self._hysteron_info['branch_ids'][i]
                    break
            else:
                raise ValueError('Cannot determine state of hysteron')
            self._hysteron_state_id_graphic.set_position(hysteron_state_graphic_position)
            self._hysteron_state_id_graphic.xy = hysteron_state_graphic_position
            self._hysteron_state_id_graphic.set_text(state_id)
            self._hysteron_state_id_graphic.set_fontsize(
                0.65 / min(2, len(state_id)) * self._aa.hysteron_state_label_size)
            self._hysteron_state_bg_graphic.set_xdata([hysteron_state_graphic_position[0]])
            self._hysteron_state_bg_graphic.set_ydata([hysteron_state_graphic_position[1]])
            if color is not None:
                self._hysteron_state_bg_graphic.set_markeredgecolor(color)


class AssemblyDrawing(Drawing):

    def __init__(self, ax: plt.Axes, _assembly: Assembly, characteristic_length, assembly_appearance,
                 element_color_handler=None, element_opacity_handler=None):
        super().__init__(ax, assembly_appearance)
        self._assembly = _assembly
        self._el_color_handler = element_color_handler
        self._el_opacity_handler = element_opacity_handler
        self._characteristic_length = characteristic_length

        # CREATE GRAPHICS FOR ASSEMBLY DRAWING
        self._node_drawings, self._element_drawings = self._make()

    def _make(self) -> tuple[list[NodeDrawing], list[ElementDrawing]]:
        node_drawings = []
        element_drawings = []
        for _element in self._assembly.get_elements():
            element_drawings.append(ElementDrawing(self._ax, _element,
                                                   0.15 * self._characteristic_length,
                                                   self._aa,
                                                   self._el_color_handler,
                                                   self._el_opacity_handler,
                                                   ))
        for _node in self._assembly.get_nodes():
            node_drawings.append(NodeDrawing(self._ax, _node, self._aa))
        return node_drawings, element_drawings

    def update(self, *args):
        for node_drawing in self._node_drawings:
            node_drawing.update()
        for element_drawing in self._element_drawings:
            element_drawing.update()


class ForceDrawing(Drawing):
    def __init__(self, ax: plt.Axes, _node: Node, force_info: dict[str, float], vector_size, assembly_appearance,
                 color_handler=None, is_preload=False):
        super().__init__(ax, assembly_appearance)
        self._node = _node
        self._force_info = force_info
        self._vector_size = vector_size * self._aa.force_vector_scaling
        self._color_handler = color_handler
        self._is_preload = is_preload

        # CREATE GRAPHICS FOR FORCE DRAWING
        self._force_graphic = self._make()

    def _make(self) -> plt.Annotation:
        if self._force_info is not None:
            direction = self._force_info['direction']
            if self._aa.force_vector_connection == 'head':
                destination = np.array((self._node.get_x(), self._node.get_y()))
                origin = destination - self._vector_size * direction
            else:
                origin = np.array((self._node.get_x(), self._node.get_y()))
                destination = origin + self._vector_size * direction
            color = (self._color_handler.determine_property_value(self._force_info['amount'])
                     if self._color_handler is not None else self._aa.force_default_outer_color)
            facecolor = self._aa.force_inner_color if not self._is_preload else self._aa.preload_force_inner_color
            force_graphic = self._ax.annotate('',
                                              xytext=(origin[0], origin[1]),
                                              xy=(destination[0], destination[1]),
                                              verticalalignment="center",
                                              arrowprops=dict(width=4, headwidth=10, lw=1.5, headlength=10, shrink=0.1,
                                                              facecolor=to_rgba(facecolor, alpha=0.65),
                                                              edgecolor=color),
                                              zorder=2 if not self._is_preload else 1.9)
        else:
            force_graphic = None
        return force_graphic

    def update(self, *args):
        if self._force_graphic is not None:
            direction = self._force_info['direction']
            if self._aa.force_vector_connection == 'head':
                destination = np.array((self._node.get_x(), self._node.get_y()))
                origin = destination - self._vector_size * direction
            else:
                origin = np.array((self._node.get_x(), self._node.get_y()))
                destination = origin + self._vector_size * direction
            self._force_graphic.set_position((origin[0], origin[1]))
            self._force_graphic.xy = (destination[0], destination[1])
            if self._color_handler is not None:
                color = self._color_handler.determine_property_value(self._force_info['amount'])
                self._force_graphic.arrow_patch.set_edgecolor(color)
        else:
            pass


class ModelDrawing(Drawing):
    def __init__(self, ax: plt.Axes, _model: Model, assembly_appearance: AssemblyAppearanceOptions,
                 characteristic_length=None, assembly_span=None,
                 element_color_handler=None, element_opacity_handler=None,
                 force_color_handler=None, force_amounts: dict = None,
                 force_vector_after_preloading=None, preforce_amounts: dict = None
                 ):
        super().__init__(ax, assembly_appearance)

        if characteristic_length is None:
            characteristic_length = _model.get_assembly().compute_characteristic_length()
        if assembly_span is None:
            xmin, ymin, xmax, ymax = _model.get_assembly().get_dimensional_bounds()
            assembly_span = max(xmax - xmin, ymax - ymin)

        self._force_amounts = (force_amounts if force_amounts is not None
                               else {n: None for n in _model.get_loaded_nodes()})

        self._initial_force_vector = (force_vector_after_preloading if force_vector_after_preloading is not None
                                      else _model.get_preforce_vector())

        self._preforce_amounts = (preforce_amounts if preforce_amounts is not None
                                  else {n: None for n in _model.get_preloaded_nodes()})

        self._assembly = _model.get_assembly()
        self._element_color_handler = element_color_handler
        self._element_opacity_handler = element_opacity_handler
        self._force_color_handler = force_color_handler
        self._node_to_dof_indices = {}
        self._characteristic_length = characteristic_length
        self._assembly_span = assembly_span
        self._loaded_nodes = _model.get_loaded_nodes()
        self._preloaded_nodes = _model.get_preloaded_nodes()
        node_nb_to_dof_indices = self._assembly.get_nodes_dof_indices()
        for _node in self._loaded_nodes | self._preloaded_nodes:
            self._node_to_dof_indices[_node] = node_nb_to_dof_indices[_node.get_node_nb()]

        self._force_directions = {}
        final_force_vector = _model.get_force_vector()
        for loaded_node in self._loaded_nodes:
            final_force = final_force_vector[self._node_to_dof_indices[loaded_node]]
            final_force_norm = np.linalg.norm(final_force)
            direction = final_force / final_force_norm
            if np.isnan(direction).any():
                direction = None
            self._force_directions[loaded_node] = direction

        self._preforce_directions = {}
        for preloaded_node in self._preloaded_nodes:
            initial_force = self._initial_force_vector[self._node_to_dof_indices[preloaded_node]]
            initial_force_norm = np.linalg.norm(initial_force)
            direction = initial_force / initial_force_norm
            if np.isnan(direction).any() or (self._aa.hide_low_preloading_forces
                                             and initial_force_norm < self._aa.low_preloading_force_threshold):
                direction = None
            self._preforce_directions[preloaded_node] = direction

        self._all_forces_info = {}
        if self._aa.show_forces:
            for _node in self._loaded_nodes:
                if self._force_directions[_node] is not None:
                    self._all_forces_info[_node] = {'direction': self._force_directions[_node],
                                                    'amount': self._force_amounts[_node]}
                else:
                    self._all_forces_info[_node] = None

        self._all_preforces_info = {}
        if self._aa.show_forces:
            for _node in self._preloaded_nodes:
                if self._preforce_directions[_node] is not None:
                    self._all_preforces_info[_node] = {'direction': self._preforce_directions[_node],
                                                       'amount': self._preforce_amounts[_node]}
                else:
                    self._all_preforces_info[_node] = None

        # CREATE GRAPHIC FOR MODEL DRAWING
        self._assembly_drawing, self._force_drawings = self._make()

    def _make(self):
        assembly_drawing = AssemblyDrawing(self._ax, self._assembly, self._characteristic_length, self._aa,
                                           self._element_color_handler, self._element_opacity_handler)
        force_drawings = set()
        if self._aa.show_forces:
            for _node in self._loaded_nodes:
                force_drawings.add(ForceDrawing(self._ax, _node, self._all_forces_info[_node],
                                                0.1 * self._assembly_span, self._aa, self._force_color_handler,
                                                is_preload=False))
            for _node in self._preloaded_nodes:
                force_drawings.add(ForceDrawing(self._ax, _node, self._all_preforces_info[_node],
                                                0.1 * self._assembly_span, self._aa, self._force_color_handler,
                                                is_preload=True))

        return assembly_drawing, force_drawings

    def update(self, *args):
        self._assembly_drawing.update()

        if self._aa.show_forces:
            for loaded_node in self._loaded_nodes:
                if self._all_forces_info[loaded_node] is not None:
                    self._all_forces_info[loaded_node]['direction'] = self._force_directions[loaded_node]
                    self._all_forces_info[loaded_node]['amount'] = self._force_amounts[loaded_node]
            for preloaded_node in self._preloaded_nodes:
                if self._all_preforces_info[preloaded_node] is not None:
                    self._all_preforces_info[preloaded_node]['direction'] = self._preforce_directions[preloaded_node]
                    self._all_preforces_info[preloaded_node]['amount'] = self._preforce_amounts[preloaded_node]

        # updating the force drawings
        for force_drawing in self._force_drawings:
            force_drawing.update()
