from ..simulation.node import Node
from ..simulation.element import Element
from ..simulation.mechanical_behavior import BivariateBehavior
from ..simulation.assembly import Assembly
from ..simulation.model import Model
from ..simulation import shape
from .graphic_settings import AssemblyAppearance as AA
from .visual_helpers import compute_zigzag_line, compute_arc_line
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class Drawing:
    def __init__(self, ax: plt.Axes):
        self._ax = ax

    def _make(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class NodeDrawing(Drawing):

    def __init__(self, ax: plt.Axes, _node: Node, is_loaded=False):
        super().__init__(ax)
        self._node = _node
        self._is_loaded = is_loaded

        # CREATE GRAPHICS FOR DRAWING
        self._text, self._node_graphic = self._make()

    def _make(self):
        if self._node.is_fixed_horizontally() and self._node.is_fixed_vertically():
            marker = 's'
        elif self._node.is_fixed_horizontally():
            marker = '<'
        elif self._node.is_fixed_vertically():
            marker = '^'
        else:
            marker = 'o'
        markersize = AA.node_size
        markercolor = AA.node_color
        text = None
        if AA.show_node_numbers:
            text = self._ax.text(self._node.get_x(), self._node.get_y(), f'{self._node.get_node_nb()}', color=AA.node_nb_color)
        graphic = self._ax.plot([self._node.get_x()], [self._node.get_y()], zorder=2.0,
                                marker=marker,
                                markersize=markersize,
                                color=markercolor)[0]
        return text, graphic

    def update(self):
        self._node_graphic.set_xdata([self._node.get_x()])
        self._node_graphic.set_ydata([self._node.get_y()])
        if AA.show_node_numbers:
            self._text.set_x(self._node.get_x())
            self._text.set_y(self._node.get_y())

class ElementDrawing(Drawing):

    def __init__(self, ax: plt.Axes, _element: Element, width, color_handler=None, opacity_handler=None):
        super().__init__(ax)
        self._element = _element
        self._color_handler = color_handler
        self._opacity_handler = opacity_handler
        self._width = width
        behavior = self._element.get_behavior()
        self._hysteron_info = behavior.get_hysteron_info() if isinstance(behavior, BivariateBehavior) else {}
        # CREATE GRAPHICS FOR DRAWING
        self._element_graphic, self._hysteron_state_bg_graphic, self._hysteron_state_id_graphic = self._make()

    def _make(self):
        if self._color_handler is None:
            color = None  # to specify later
        else:
            color = self._color_handler.determine_property_value(self._element)

        if self._opacity_handler is None or not isinstance(self._element.get_shape(), shape.DistancePointLine):
            opacity = None  # to specify later
        else:
            opacity = self._opacity_handler.determine_property_value(self._element)

        hysteron_state_drawing_position = None
        if isinstance(self._element.get_shape(), shape.Segment):
            color = color if color is not None else AA.spring_default_color
            opacity = opacity if opacity is not None else AA.spring_default_opacity
            x0, y0, x1, y1 = self._element.get_shape().get_nodal_coordinates()
            x_coords, y_coords = compute_zigzag_line((x0, y0), (x1, y1), 8 + 2 * AA.nb_spring_coils,
                                                     self._width * AA.spring_width_scaling)
            node0, node1 = self._element.get_shape().get_nodes()
            # if abs(node0.get_node_nb() - node1.get_node_nb()) > 1.5:
            #     y_coords += .3 if node0.get_node_nb() % 3 == 0 else -.3
            if AA.show_state_of_hysterons and self._hysteron_info:
                hysteron_state_drawing_position = ((x0 + x1) / 2, (y0 + y1) / 2)
            element_graphic = \
            self._ax.plot(x_coords, y_coords, lw=AA.spring_linewidth, color=color, alpha=opacity, zorder=0.1)[0]
        elif isinstance(self._element.get_shape(), shape.Angle):
            color = color if color is not None else AA.rotation_spring_default_color
            opacity = opacity if opacity is not None else AA.rotation_spring_default_opacity
            x0, y0, x1, y1, x2, y2 = self._element.get_shape().get_nodal_coordinates()
            center = (x1, y1)
            angle = shape.Angle.calculate_angle(x0, y0, x1, y1, x2, y2)
            end_angle = shape.Angle.calculate_angle(x1 + 1, y1, x1, y1, x2, y2)
            start_angle = end_angle - angle
            x_coords, y_coords = compute_arc_line(center, self._width * AA.rotation_spring_radius_scaling, start_angle,
                                                  end_angle)
            if AA.show_state_of_hysterons and self._hysteron_info:
                mid_angle = (start_angle + end_angle) / 2
                hysteron_state_drawing_position = (
                    center[0] + self._width * AA.rotation_spring_radius_scaling * np.cos(mid_angle),
                    center[0] + self._width * AA.rotation_spring_radius_scaling * np.sin(mid_angle))
            element_graphic = \
            self._ax.plot(x_coords, y_coords, lw=AA.rotation_spring_linewidth, color=color, alpha=opacity, zorder=0)[0]
        elif isinstance(self._element.get_shape(), shape.Area):
            color = color if color is not None else AA.area_spring_default_color
            opacity = opacity if opacity is not None else AA.area_spring_default_opacity
            coordinates = self._element.get_shape().get_nodal_coordinates()
            x, y = coordinates[::2], coordinates[1::2]
            zorder = np.min(x) / np.abs(np.max(x) - np.min(x))
            element_graphic = self._ax.fill(x, y, color=color, alpha=opacity, zorder=zorder)[0]
            if AA.show_state_of_hysterons and self._hysteron_info:
                hysteron_state_drawing_position = (np.mean(x), np.mean(y))
        elif isinstance(self._element.get_shape(), shape.Path):
            color = color if color is not None else AA.line_spring_default_color
            opacity = opacity if opacity is not None else AA.line_spring_default_opacity
            nodes = [_node for _shape in self._element.get_shape().get_shapes() for _node in _shape.get_nodes()]
            coordinates = []
            for i, _node in enumerate(nodes):
                if i % 2 == 0 or i == len(nodes) - 1:
                    coordinates += [_node.get_x(), _node.get_y()]
            x, y = coordinates[::2], coordinates[1::2]
            lengths = [0.0]
            for i in range(len(x) - 1):
                lengths.append(lengths[-1] + shape.Segment.calculate_length(x[i], y[i], x[i + 1], y[i + 1]))
            t = np.linspace(0, 1, 10) * lengths[-1]
            xx = interp1d(lengths, x)(t[1:-1])
            yy = interp1d(lengths, y)(t[1:-1])
            element_graphic0 = self._ax.plot(x, y, lw=AA.line_spring_linewidth, color=color, alpha=opacity, zorder=0)[0]
            element_graphic1 = \
            self._ax.plot(xx, yy, ls='', marker='o', markersize=AA.line_spring_linewidth * 0.4, color='#CECECE',
                          zorder=0.1)[0]
            element_graphic = (element_graphic0, element_graphic1)
            if AA.show_state_of_hysterons and self._hysteron_info:
                hysteron_state_drawing_position = (np.mean(x), np.mean(y))
        elif isinstance(self._element.get_shape(), (shape.DistancePointLine, shape.SquaredDistancePointSegment)):
            color = color if color is not None else AA.distance_spring_line_default_color
            opacity = opacity if opacity is not None else AA.distance_spring_line_default_opacity
            node0, _, _ = self._element.get_shape().get_nodes()
            x0, y0, x1, y1, x2, y2 = self._element.get_shape().get_nodal_coordinates()
            element_graphic0 = self._ax.plot(x0, y0, zorder=2.1,
                                             marker='.',
                                             markersize=AA.node_size / 2.0,
                                             alpha=opacity,
                                             color=color)[0]
            element_graphic1 = self._ax.plot([x1, x2], [y1, y2], lw=AA.distance_spring_line_linewidth,
                                             color=color, alpha=opacity, zorder=0)[0]
            element_graphic2 = self._ax.plot([x1, x2], [y1, y2], lw=0.75,
                                             color=AA.distance_spring_line_default_color, zorder=0)[0]
            element_graphic = (element_graphic0, element_graphic1, element_graphic2)
        else:
            raise NotImplementedError('Cannot draw element because no implementation of how to draw its shape')

        if AA.show_state_of_hysterons and self._hysteron_info:
            internal_coord = self._element.get_internal_coordinates()
            for i, interval in enumerate(self._hysteron_info['branch_intervals']):
                if interval[0] <= internal_coord <= interval[1]:
                    state_id = self._hysteron_info['branch_ids'][i]
                    break
            else:
                raise ValueError('Cannot determine state of hysteron')
            hysteron_state_bg_graphic = self._ax.plot(*hysteron_state_drawing_position, 'o', zorder=1.0,
                                                      markersize=AA.hysteron_state_label_size,
                                                      color=AA.hysteron_state_bg_color, markeredgecolor=color,
                                                      markeredgewidth=AA.spring_linewidth)[0]
            hysteron_state_id_graphic = self._ax.annotate(state_id, xy=hysteron_state_drawing_position,
                                                          color=AA.hysteron_state_txt_color,
                                                          fontsize=0.65 / min(2,
                                                                              len(state_id)) * AA.hysteron_state_label_size,
                                                          weight='bold',
                                                          verticalalignment="center",
                                                          horizontalalignment="center",
                                                          zorder=1.5)
        else:
            hysteron_state_bg_graphic = None
            hysteron_state_id_graphic = None
        return element_graphic, hysteron_state_bg_graphic, hysteron_state_id_graphic

    def update(self):
        hysteron_state_graphic_position = None
        if isinstance(self._element.get_shape(), shape.Segment):
            x0, y0, x1, y1 = self._element.get_shape().get_nodal_coordinates()
            x_coords, y_coords = compute_zigzag_line((x0, y0), (x1, y1), 8 + 2 * AA.nb_spring_coils,
                                                     self._width * AA.spring_width_scaling)
            node0, node1 = self._element.get_shape().get_nodes()
            # if abs(node0.get_node_nb() - node1.get_node_nb()) > 1.5:
            #     y_coords += .3 if node0.get_node_nb() % 3 == 0 else -.3
            self._element_graphic.set_xdata(x_coords)
            self._element_graphic.set_ydata(y_coords)
            if self._hysteron_info and AA.show_state_of_hysterons:
                hysteron_state_graphic_position = ((x0 + x1) / 2, (y0 + y1) / 2)

        elif isinstance(self._element.get_shape(), shape.Angle):
            x0, y0, x1, y1, x2, y2 = self._element.get_shape().get_nodal_coordinates()
            center = (x1, y1)
            angle = shape.Angle.calculate_angle(x0, y0, x1, y1, x2, y2)
            end_angle = shape.Angle.calculate_angle(x1 + 1, y1, x1, y1, x2, y2)
            start_angle = end_angle - angle
            x_coords, y_coords = compute_arc_line(center, self._width * AA.rotation_spring_radius_scaling, start_angle,
                                                  end_angle)
            self._element_graphic.set_xdata(x_coords)
            self._element_graphic.set_ydata(y_coords)
            if self._hysteron_info and AA.show_state_of_hysterons:
                mid_angle = (start_angle + end_angle) / 2
                hysteron_state_graphic_position = (
                    center[0] + self._width * AA.rotation_spring_radius_scaling * np.cos(mid_angle),
                    center[1] + self._width * AA.rotation_spring_radius_scaling * np.sin(mid_angle))
        elif isinstance(self._element.get_shape(), shape.Area):
            coordinates = self._element.get_shape().get_nodal_coordinates()
            self._element_graphic.set_xy(np.array(coordinates).reshape(-1, 2))
            if self._hysteron_info and AA.show_state_of_hysterons:
                hysteron_state_graphic_position = (np.mean(coordinates[::2]), np.mean(coordinates[1::2]))
        elif isinstance(self._element.get_shape(), shape.Path):
            element_graphic0, element_graphic1 = self._element_graphic
            coordinates = self._element.get_shape().get_nodal_coordinates()
            nodes = [_node for _shape in self._element.get_shape().get_shapes() for _node in _shape.get_nodes()]
            coordinates = []
            for i, _node in enumerate(nodes):
                if i % 2 == 0 or i == len(nodes) - 1:
                    coordinates += [_node.get_x(), _node.get_y()]
            x, y = coordinates[::2], coordinates[1::2]
            lengths = [0.0]
            for i in range(len(x) - 1):
                lengths.append(lengths[-1] + shape.Segment.calculate_length(x[i], y[i], x[i + 1], y[i + 1]))
            t = np.linspace(0, 1, 10) * lengths[-1]
            xx = interp1d(lengths, x)(t[1:-1])
            yy = interp1d(lengths, y)(t[1:-1])
            element_graphic0.set_xdata(x)
            element_graphic0.set_ydata(y)
            element_graphic1.set_xdata(xx)
            element_graphic1.set_ydata(yy)
            if self._hysteron_info and AA.show_state_of_hysterons:
                hysteron_state_graphic_position = (np.mean(coordinates[::2]), np.mean(coordinates[1::2]))
        elif isinstance(self._element.get_shape(), (shape.DistancePointLine, shape.SquaredDistancePointSegment)):
            element_graphic0, element_graphic1, element_graphic2 = self._element_graphic
            x0, y0, x1, y1, x2, y2 = self._element.get_shape().get_nodal_coordinates()
            element_graphic0.set_xdata(x0)
            element_graphic0.set_ydata(y0)
            element_graphic1.set_xdata([x1, x2])
            element_graphic1.set_ydata([y1, y2])
            element_graphic2.set_xdata([x1, x2])
            element_graphic2.set_ydata([y1, y2])
        else:
            raise NotImplementedError(
                'Cannot update element drawing, because no implementation of how to draw its shape')
        color = None
        if self._color_handler is not None:
            color = self._color_handler.determine_property_value(self._element)
            if isinstance(self._element.get_shape(), (shape.DistancePointLine, shape.SquaredDistancePointSegment)):
                graphic0, graphic1, _ = self._element_graphic
                graphic0.set_color(color)
                graphic1.set_color(color)
            elif isinstance(self._element.get_shape(), shape.Path):
                graphic0, _ = self._element_graphic
                graphic0.set_color(color)
            else:
                self._element_graphic.set_color(color)
        if self._opacity_handler is not None and isinstance(self._element.get_shape(), shape.DistancePointLine):
            opacity = self._opacity_handler.determine_property_value(self._element)
            graphic0, graphic1, _ = self._element_graphic
            graphic0.set_alpha(opacity)
            graphic1.set_alpha(opacity)

        if self._hysteron_state_bg_graphic is not None and self._hysteron_state_id_graphic is not None:
            self._hysteron_state_bg_graphic.set_xdata(hysteron_state_graphic_position[0])
            self._hysteron_state_bg_graphic.set_ydata(hysteron_state_graphic_position[1])
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
            self._hysteron_state_id_graphic.set_fontsize(0.65 / min(2, len(state_id)) * AA.hysteron_state_label_size)
            self._hysteron_state_bg_graphic.set_xdata(hysteron_state_graphic_position[0])
            self._hysteron_state_bg_graphic.set_ydata(hysteron_state_graphic_position[1])
            if color is not None:
                self._hysteron_state_bg_graphic.set_markeredgecolor(color)


class AssemblyDrawing(Drawing):

    def __init__(self, ax: plt.Axes, _assembly: Assembly, characteristic_length, element_color_handler=None,
                 element_opacity_handler=None):
        super().__init__(ax)
        self._assembly = _assembly
        self._el_color_handler = element_color_handler
        self._el_opacity_handler = element_opacity_handler
        self._characteristic_length = characteristic_length

        # CREATE GRAPHICS FOR ASSEMBLY DRAWING
        self._node_drawings, self._element_drawings = self._make()

    def _make(self) -> tuple[set[NodeDrawing], set[ElementDrawing]]:
        node_drawings = set()
        element_drawings = set()
        for _node in self._assembly.get_nodes():
            node_drawings.add(NodeDrawing(self._ax, _node))
        for _element in self._assembly.get_elements():
            element_drawings.add(ElementDrawing(self._ax, _element,
                                                0.15 * self._characteristic_length,
                                                self._el_color_handler,
                                                self._el_opacity_handler))
        return node_drawings, element_drawings

    def update(self):
        for node_drawing in self._node_drawings:
            node_drawing.update()
        for element_drawing in self._element_drawings:
            element_drawing.update()


class ForceDrawing(Drawing):
    def __init__(self, ax: plt.Axes, _node: Node, force_info: dict[str, float], vector_size, color_handler=None):
        super().__init__(ax)
        self._node = _node
        self._force_info = force_info
        self._vector_size = vector_size
        self._color_handler = color_handler

        # CREATE GRAPHICS FOR FORCE DRAWING
        self._force_graphic = self._make()

    def _make(self) -> plt.Annotation:
        origin = np.array((self._node.get_x(), self._node.get_y()))
        direction = self._force_info['direction']
        destination = origin + self._vector_size * direction
        color = self._color_handler.determine_property_value(self._force_info['magnitude']) \
            if self._color_handler is not None else AA.force_default_outer_color
        force_graphic = self._ax.annotate('',
                                          xytext=(origin[0], origin[1]),
                                          xy=(destination[0], destination[1]),
                                          verticalalignment="center",
                                          arrowprops=dict(width=4, headwidth=10, shrink=0.1, lw=1.5,
                                                          facecolor=AA.force_inner_color, edgecolor=color),
                                          zorder=2)

        return force_graphic

    def update(self):
        origin = np.array((self._node.get_x(), self._node.get_y()))
        direction = self._force_info['direction']
        destination = origin + self._vector_size * direction
        self._force_graphic.set_position((origin[0], origin[1]))
        self._force_graphic.xy = (destination[0], destination[1])
        if self._color_handler is not None:
            color = self._color_handler.determine_property_value(self._force_info['magnitude'])
            self._force_graphic.arrow_patch.set_edgecolor(color)


class ModelDrawing(Drawing):
    def __init__(self, ax: plt.Axes, _model: Model,
                 external_force_vector=None, characteristic_length=None, assembly_span=None,
                 element_color_handler=None, element_opacity_handler=None, force_color_handler=None):
        super().__init__(ax)
        if characteristic_length is None:
            characteristic_length = _model.get_assembly().compute_characteristic_length()
        if assembly_span is None:
            xmin, ymin, xmax, ymax = _model.get_assembly().get_dimensional_bounds()
            assembly_span = max(xmax - xmin, ymax - ymin)
        if external_force_vector is None:
            external_force_vector = _model.get_force_vector()
        self._assembly = _model.get_assembly()
        self._element_color_handler = element_color_handler
        self._element_opacity_handler = element_opacity_handler
        self._force_color_handler = force_color_handler
        self._external_force_vector = external_force_vector
        self._loaded_nodes_to_dof_indices = {}
        self._characteristic_length = characteristic_length
        self._assembly_span = assembly_span
        node_nb_to_dof_indices = self._assembly.get_nodes_dof_indices()
        for loaded_node in _model.get_loaded_nodes():
            self._loaded_nodes_to_dof_indices[loaded_node] = node_nb_to_dof_indices[loaded_node.get_node_nb()]

        self._all_loaded_node_info = {}
        self._final_force_vector = _model.get_force_vector()
        for loaded_node in _model.get_loaded_nodes():
            ldm_force = self._final_force_vector[self._loaded_nodes_to_dof_indices[loaded_node]]
            force = self._external_force_vector[self._loaded_nodes_to_dof_indices[loaded_node]]
            direction = ldm_force / np.linalg.norm(ldm_force)
            magnitude = np.linalg.norm(force)
            self._all_loaded_node_info[loaded_node] = {'direction': direction, 'magnitude': magnitude}

        # CREATE GRAPHIC FOR MODEL DRAWING
        self._assembly_drawing, self._force_drawings = self._make()

    def _make(self):
        assembly_drawing = AssemblyDrawing(self._ax, self._assembly, self._characteristic_length,
                                           self._element_color_handler, self._element_opacity_handler)
        force_drawings = set()
        if AA.show_forces:
            for _node, force_info in self._all_loaded_node_info.items():
                force_drawings.add(
                    ForceDrawing(self._ax, _node, force_info, 0.1 * self._assembly_span,
                                 self._force_color_handler))

        return assembly_drawing, force_drawings

    def update(self):
        self._assembly_drawing.update()

        # First, updating the external force vector of each loaded node
        for loaded_node, force_info in self._all_loaded_node_info.items():
            ldm_force = self._final_force_vector[self._loaded_nodes_to_dof_indices[loaded_node]]
            force = self._external_force_vector[self._loaded_nodes_to_dof_indices[loaded_node]]
            direction = ldm_force / np.linalg.norm(ldm_force)
            magnitude = np.linalg.norm(force)
            self._all_loaded_node_info[loaded_node]['direction'] = direction
            self._all_loaded_node_info[loaded_node]['magnitude'] = magnitude

        # Only then, updating the force drawings
        for force_drawing in self._force_drawings:
            force_drawing.update()
